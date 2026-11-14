"""VIS header detection.

Watches the instantaneous-frequency stream for the SSTV calibration
header.  The detection pattern is the part of the header that matters:

    1900 Hz leader (>=120 ms visible)  ->  1200 Hz start bit (30 ms)
    ->  7 data bits + even parity (30 ms each, 1100 Hz = 1, 1300 Hz = 0)
    ->  1200 Hz stop bit (30 ms)  ->  image data

Requiring a long 1900 Hz leader, a valid start bit, eight valid FSK bits,
correct even parity, a low stop bit and a *known* VIS code makes false
detections on voice or noise essentially impossible, while still catching
headers that began before the stream did (the first calibration leader
and 10 ms break are not required).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sstv.freq import ToneFrame, moving_average, refine_falling_edge
from sstv.modes import (FREQ_LEADER, FREQ_VIS_BIT0, FREQ_VIS_BIT1,
                        SSTVMode, mode_for_vis)

log = logging.getLogger(__name__)

LEADER_LO_HZ = 1740.0
LEADER_HI_HZ = 2060.0
START_DROP_HZ = 1500.0    # start bit: leader drops below this and stays
START_DROP_MIN_MS = 20.0  # sustained drop; the 10 ms header break must not
                          # qualify (the real start bit run extends through
                          # the VIS bits and lasts >300 ms)
MIN_LEADER_S = 0.120      # visible leader length required
LEADER_GAP_S = 0.004      # tolerated gap inside a leader region
SMOOTH_MS = 3.0           # smoothing for band decisions (noise robustness)
BIT_MS = 30.0
N_DATA_BITS = 7
N_BITS = 8                  # 7 data + 1 even parity
BIT_EYE_MS = 8.0            # +/- sampling window inside each bit
BIT_TOL_HZ = 110.0
MIN_LEADER_DENSITY = 0.5     # in-band fraction required inside a leader region
LEADER_END_WINDOW_MS = 45.0  # leader must end this close before the start bit
FIRST_SYNC_MAX_HZ = 1300.0   # image data starts with a 1200 Hz sync pulse
BLACKOUT_S = 0.35         # skip past a failed candidate (its bits + margin)
BUFFER_S = 2.5              # rolling history kept for pattern matching


@dataclass
class Detection:
    mode: SSTVMode
    data_start: int  # global sample index where image lines begin


class VISDetector:
    """Streaming VIS header detector with a small rolling history."""

    def __init__(self, fs: float):
        self.fs = float(fs)
        self._buf = np.zeros(0)          # raw instantaneous frequency
        self._gate = np.zeros(0, bool)   # signal-present mask
        self._buf_start = 0              # global index of _buf[0]
        self._env_ref = 0.0
        self._in_vis = False
        self._t_bit: Optional[int] = None
        self._retry_after = 0            # global index: skip candidates before it

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._in_vis = False
        self._t_bit = None

    def _gate_frame(self, frame: ToneFrame) -> np.ndarray:
        """Signal-present mask: rejects dead air, not quiet signals.

        The reference tracks the *current* level within a couple of frames
        (a slow-decaying reference kept the gate closed for many seconds
        after any loud transmission - e.g. a voice callsign - and silenced
        a following SSTV image that was transmitted tens of dB quieter).
        The VIS validation chain is what rejects noise; this mask only
        needs to kill digital silence.
        """
        env = frame.env
        if env.size == 0:
            return env.astype(bool)
        local = (float(np.percentile(env, 95)) if env.size > 16
                 else float(np.max(env)))
        self._env_ref = 0.3 * self._env_ref + 0.7 * local
        return env > max(0.05 * self._env_ref, 1e-5)

    def _append(self, frame: ToneFrame) -> None:
        if len(self._buf) == 0:
            self._buf_start = frame.start
        gap = frame.start - (self._buf_start + len(self._buf))
        if gap > 0:
            self._buf = np.concatenate([self._buf, np.full(gap, np.nan)])
            self._gate = np.concatenate([self._gate, np.zeros(gap, bool)])
        self._buf = np.concatenate([self._buf, frame.freq])
        self._gate = np.concatenate([self._gate, self._gate_frame(frame)])
        # bound the history
        max_len = int(BUFFER_S * self.fs)
        if len(self._buf) > max_len:
            drop = len(self._buf) - max_len
            self._buf = self._buf[drop:]
            self._gate = self._gate[drop:]
            self._buf_start += drop

    def _mean(self, i0: int, i1: int) -> float:
        a = max(i0, self._buf_start)
        b = min(i1, self._buf_start + len(self._buf))
        if b <= a:
            return float("nan")
        return float(np.mean(self._buf[a - self._buf_start:b - self._buf_start]))

    # ------------------------------------------------------------------
    def process(self, frame: ToneFrame) -> Optional[Detection]:
        if frame.freq.size == 0:
            return None
        self._append(frame)
        if self._in_vis:
            return self._vis()
        return self._seek()

    # ------------------------------------------------------------------
    def _seek(self) -> Optional[Detection]:
        fs = self.fs
        n = len(self._buf)
        smooth = moving_average(self._buf, max(1, int(round(SMOOTH_MS * fs / 1000.0))))
        in_leader = (smooth > LEADER_LO_HZ) & (smooth < LEADER_HI_HZ) & self._gate
        below = (smooth < START_DROP_HZ) & self._gate

        min_leader = int(round(MIN_LEADER_S * fs))
        gap_max = int(round(LEADER_GAP_S * fs))
        drop_min = int(round(START_DROP_MIN_MS * fs / 1000.0))
        bit_len = int(round(BIT_MS * fs / 1000.0))
        eye = int(round(BIT_EYE_MS * fs / 1000.0))

        # Leader regions: maximal spans where in-band samples dominate and
        # gaps stay small (noise-tolerant).  Indices are LOCAL.
        gap_max = int(round(LEADER_GAP_S * fs))
        leader_end_win = int(round(LEADER_END_WINDOW_MS * fs / 1000.0))
        for l_start, l_end, count in self._leader_regions(in_leader, gap_max):
            if count < min_leader:
                continue
            if count / max(l_end - l_start, 1) < MIN_LEADER_DENSITY:
                continue  # too fragmented: probably noise
            # start bit: first sustained drop below START_DROP_HZ after the
            # leader region ends
            t_local = self._find_start_drop(below, l_end, drop_min, n)
            if t_local is None:
                continue  # not arrived yet, or no drop: retry next frame
            t_bit = self._buf_start + t_local
            if t_bit < self._retry_after:
                continue  # recently failed around here: don't re-litigate noise
            # the leader must end shortly before the start bit
            if t_local - l_end > leader_end_win:
                continue
            # VIS bits, stop bit and the first sync pulse must have arrived
            need = t_bit + (N_BITS + 2) * bit_len + eye
            if self._buf_start + n < need:
                continue
            self._t_bit = t_bit
            det = self._vis()
            if det is not None:
                self._retry_after = det.data_start
                return det
            # bits were invalid: false leader - blackout this region
            self._in_vis = False
            self._t_bit = None
            self._retry_after = t_bit + int(round(BLACKOUT_S * fs))
        return None

    def _leader_regions(self, in_leader: np.ndarray, gap_max: int):
        """Yield (start, end, in_band_count) for each leader region.

        Maximal spans of in-band samples tolerating gaps of up to
        ``gap_max``.  Vectorized via the True-sample indices: this runs
        on every chunk against the whole rolling buffer, so a Python
        sample loop here would dominate the CPU budget.
        """
        idx = np.nonzero(in_leader)[0]
        if idx.size == 0:
            return []
        # split where the gap between consecutive True samples is too large
        # (gap = diff - 1 False samples; the old walk broke at gap > gap_max)
        breaks = np.nonzero(np.diff(idx) > gap_max)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [idx.size]])
        regions = []
        for s, e in zip(starts, ends):
            seg = idx[s:e]
            regions.append((int(seg[0]), int(seg[-1]) + 1, int(seg.size)))
        return regions

    def _find_start_drop(self, below: np.ndarray, l_end: int,
                         drop_min: int, n: int) -> Optional[int]:
        """First sustained low run that starts after a leader region."""
        idx = np.nonzero(below[l_end:])[0]
        if idx.size == 0:
            return None
        idx = idx + l_end
        # group consecutive indices into runs
        breaks = np.nonzero(np.diff(idx) > 1)[0]
        run_starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        run_ends = np.concatenate([idx[breaks] + 1, [idx[-1] + 1]])
        lengths = run_ends - run_starts
        ok = np.nonzero(lengths >= drop_min)[0]
        if ok.size == 0:
            return None
        r0 = ok[0]
        start, end = int(run_starts[r0]), int(run_ends[r0])
        if end >= n:
            return None  # run may still grow; wait for more data
        return start

    # ------------------------------------------------------------------
    def _vis(self) -> Optional[Detection]:
        fs = self.fs
        t_bit = self._t_bit
        bit_len = int(round(BIT_MS * fs / 1000.0))
        eye = int(round(BIT_EYE_MS * fs / 1000.0))
        if self._buf_start + len(self._buf) < t_bit + (N_BITS + 1) * bit_len + eye:
            self._in_vis = True
            return None  # wait for the rest of the bits

        bits: List[int] = []
        for k in range(N_BITS):
            # bit k occupies [t_bit + (k+1)*bit_len, t_bit + (k+2)*bit_len)
            centre = t_bit + (k + 1.5) * bit_len
            mean = self._mean(int(centre) - eye, int(centre) + eye)
            if not np.isfinite(mean):
                self.reset()
                return None
            d1 = abs(mean - FREQ_VIS_BIT1)
            d0 = abs(mean - FREQ_VIS_BIT0)
            if min(d1, d0) > BIT_TOL_HZ:
                self.reset()
                return None
            bits.append(1 if d1 < d0 else 0)

        data_bits = bits[:N_DATA_BITS]
        parity = bits[N_DATA_BITS]
        code = sum(b << k for k, b in enumerate(data_bits))
        if (sum(data_bits) + parity) % 2 != 0:
            log.debug("VIS parity error (code 0x%02x), rejected", code)
            self.reset()
            return None
        stop = self._mean(t_bit + (N_BITS + 1) * bit_len,
                          t_bit + (N_BITS + 2) * bit_len)
        if not np.isfinite(stop) or stop > 1500.0:
            self.reset()
            return None
        # the image body starts with a 1200 Hz sync pulse: a strong final
        # check that rejects noise-shaped false positives
        data_start = t_bit + (N_BITS + 2) * bit_len
        sync = self._mean(data_start + int(round(1.5 * fs / 1000.0)),
                          data_start + int(round(7.5 * fs / 1000.0)))
        if not np.isfinite(sync) or sync > FIRST_SYNC_MAX_HZ:
            self.reset()
            return None
        # drop the consumed header from the buffer so it cannot re-trigger
        drop = data_start - self._buf_start
        if drop > 0:
            drop = min(drop, len(self._buf))
            self._buf = self._buf[drop:]
            self._gate = self._gate[drop:]
            self._buf_start += drop
        self.reset()

        mode = mode_for_vis(code)
        if mode is None:
            log.info("Unsupported SSTV VIS code 0x%02x (%d)", code, code)
            return None
        log.info("VIS detected: %s (code 0x%02x), image data at sample %d",
                 mode.name, code, data_start)
        return Detection(mode=mode, data_start=data_start)

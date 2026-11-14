"""Robot 36 SSTV line decoder.

Robot 36 (VIS code 8) transmits a 320x240 image in ~36 s.  Every 150 ms
line carries its own luminance plus alternating chroma:

    1200 Hz  9 ms    horizontal sync
    1500 Hz  3 ms    sync porch
    Y scan   88 ms   320 pixels, 1500 Hz (black) .. 2300 Hz (white)
    sep      4.5 ms  1500 Hz -> this line's chroma is Cr (R-Y)
                      2300 Hz -> this line's chroma is Cb (B-Y)
    1900 Hz  1.5 ms  chroma porch
    C scan   44 ms   320 chroma samples

The decoder re-anchors each line on its sync pulse, so clock drift and
dropouts only affect the lines they touch.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from sstv.encoder import (C_SCAN_MS, CHROMA_PORCH_MS, HEIGHT, LINE_MS,
                          PORCH_MS, SEP_MS, SYNC_MS, WIDTH, Y_SCAN_MS,
                          ycbcr_to_rgb)
from sstv.events import EventType, SSTVEvent
from sstv.freq import ToneFrame, moving_average, refine_falling_edge
from sstv.modes import (FREQ_BLACK, FREQ_WHITE, SSTVMode, freq_to_level,
                        register_mode)

log = logging.getLogger(__name__)

SYNC_THRESH_HZ = 1350.0     # smoothed frequency below this = sync pulse
SYNC_MIN_MS = 4.5           # minimum run length that counts as sync
SYNC_SEARCH_EARLY_MS = 50.0 # how far before the expected sync to search
SYNC_SEARCH_LATE_MS = 50.0  # how far after
GIVE_UP_MS = 600.0          # no sync for this long -> image over
SEP_THRESHOLD_HZ = 1850.0   # separator above this = Cb line, below = Cr

_MS = 0.001


class Robot36Decoder:
    """Decodes one Robot 36 image from a stream of ToneFrames."""

    def __init__(self, fs: float, start_index: int):
        self.fs = float(fs)
        self.done = False
        self.truncated = False
        self.line = 0                      # next line index to fill
        self._buf = np.zeros(0)            # instantaneous-frequency samples
        self._buf_start = start_index      # global index of _buf[0]
        self._sync = None                  # global index of current line's sync
        self._syncs_found = 0
        self._expected_syncs = 0
        self._search_from = start_index
        self._search_deadline = start_index + int(GIVE_UP_MS * _MS * fs)
        self._first_search_done = False
        # image planes in byte units
        self._y = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self._cb = np.full((HEIGHT, WIDTH), 128.0, dtype=np.float32)
        self._cr = np.full((HEIGHT, WIDTH), 128.0, dtype=np.float32)
        self._have_y = np.zeros(HEIGHT, dtype=bool)
        self._have_cb = np.zeros(HEIGHT, dtype=bool)
        self._have_cr = np.zeros(HEIGHT, dtype=bool)

    # ------------------------------------------------------------------
    def _append(self, frame: ToneFrame) -> None:
        end = self._buf_start + len(self._buf)
        if len(self._buf) == 0:
            self._buf_start = max(frame.start, self._buf_start)
        if frame.start > end:
            gap = frame.start - end
            self._buf = np.concatenate([self._buf, np.zeros(gap)])
            end = frame.start
        skip = max(0, end - frame.start)
        self._buf = np.concatenate([self._buf, frame.freq[skip:]])

    def _trim(self, keep_from: int) -> None:
        drop = keep_from - self._buf_start
        if drop > 0:
            drop = min(drop, len(self._buf))
            self._buf = self._buf[drop:]
            self._buf_start += drop

    # ------------------------------------------------------------------
    def _find_sync(self, lo: int, hi: int) -> Optional[int]:
        """First sync-pulse start in the global index range [lo, hi)."""
        a = max(lo, self._buf_start)
        b = min(hi, self._buf_start + len(self._buf))
        if b <= a:
            return None
        seg = self._buf[a - self._buf_start:b - self._buf_start]
        smooth = moving_average(seg, max(1, int(round(1.0 * _MS * self.fs))))
        low = smooth < SYNC_THRESH_HZ
        min_run = int(round(SYNC_MIN_MS * _MS * self.fs))
        idx = 0
        n = len(low)
        while idx < n:
            if low[idx]:
                run = idx
                while run < n and low[run]:
                    run += 1
                if run - idx >= min_run:
                    s = a + idx
                    # refine the sync edge to the transition midpoint
                    s = a + refine_falling_edge(seg, idx, self.fs)
                    return s
                idx = run
            else:
                idx += 1
        return None

    def _mean(self, i0: int, i1: int) -> float:
        a = max(i0, self._buf_start)
        b = min(i1, self._buf_start + len(self._buf))
        if b <= a:
            return float("nan")
        return float(np.mean(self._buf[a - self._buf_start:b - self._buf_start]))

    def _sample_scan(self, start: int, span_ms: float) -> np.ndarray:
        """Mean instantaneous frequency for each of WIDTH pixels."""
        t1 = start + int(round(span_ms * _MS * self.fs))
        a = max(start, self._buf_start)
        b = min(t1, self._buf_start + len(self._buf))
        if b <= a or a > start:
            return np.full(WIDTH, np.nan)  # buffer does not cover the scan
        seg = self._buf[a - self._buf_start:b - self._buf_start]
        csum = np.concatenate([[0.0], np.cumsum(seg)])
        total = len(seg)
        edges = np.round(np.linspace(0, total, WIDTH + 1)).astype(int)
        means = (csum[edges[1:]] - csum[edges[:-1]]) / (edges[1:] - edges[:-1])
        return means

    # ------------------------------------------------------------------
    def process(self, frame: ToneFrame) -> List[SSTVEvent]:
        if self.done or frame.freq.size == 0:
            return []
        self._append(frame)
        events: List[SSTVEvent] = []

        while not self.done:
            if self._sync is None:
                found = self._find_sync(self._search_from,
                                        self._buf_start + len(self._buf))
                if found is not None:
                    self._sync = found
                    self._syncs_found += 1
                elif (self._buf_start + len(self._buf)) >= self._search_deadline:
                    self._finish(events, truncated=self.line > 0)
                    break
                else:
                    # wait for more data; keep the buffer tidy
                    self._trim(max(self._search_from - int(10 * _MS * self.fs),
                                   self._buf_start))
                    break
            need = self._sync + int(round(LINE_MS * _MS * self.fs))
            if self._buf_start + len(self._buf) < need:
                break
            events += self._decode_line(self._sync)
            self._sync = None

        return events

    # ------------------------------------------------------------------
    def _decode_line(self, sync: int) -> List[SSTVEvent]:
        events: List[SSTVEvent] = []
        fs = self.fs
        # chroma identity from the separator tone
        sep = self._mean(sync + int(round((SYNC_MS + PORCH_MS + Y_SCAN_MS + 0.5) * _MS * fs)),
                         sync + int(round((SYNC_MS + PORCH_MS + Y_SCAN_MS + SEP_MS - 0.5) * _MS * fs)))
        chroma_is_cr = not (np.isfinite(sep) and sep > SEP_THRESHOLD_HZ)

        # line index: next counter slot with matching chroma parity
        line = self.line
        if line >= HEIGHT:
            self._finish(events, truncated=False)
            return events
        if (line % 2 == 0) != chroma_is_cr:
            line += 1
        if line >= HEIGHT:
            self._finish(events, truncated=self.line > 0)
            return events
        # fill any lines we skipped (sync was lost) with the previous row
        for miss in range(self.line, line):
            if miss > 0:
                self._y[miss] = self._y[miss - 1]
                self._cb[miss] = self._cb[miss - 1]
                self._cr[miss] = self._cr[miss - 1]
                self._have_y[miss] = self._have_y[miss - 1]
                self._have_cb[miss] = self._have_cb[miss - 1]
                self._have_cr[miss] = self._have_cr[miss - 1]

        y_off = sync + int(round((SYNC_MS + PORCH_MS) * _MS * fs))
        y_vals = self._sample_scan(y_off, Y_SCAN_MS)
        c_off = sync + int(round((SYNC_MS + PORCH_MS + Y_SCAN_MS + SEP_MS + CHROMA_PORCH_MS) * _MS * fs))
        c_vals = self._sample_scan(c_off, C_SCAN_MS)

        if np.all(np.isfinite(y_vals)):
            self._y[line] = np.clip(freq_to_level(y_vals), 0, 255)
            self._have_y[line] = True
        if np.all(np.isfinite(c_vals)):
            if chroma_is_cr:
                self._cr[line] = np.clip(freq_to_level(c_vals), 0, 255)
                self._have_cr[line] = True
            else:
                self._cb[line] = np.clip(freq_to_level(c_vals), 0, 255)
                self._have_cb[line] = True
        self._expected_syncs += 1

        # emit rows: the just-decoded line, plus re-finalize its partner line
        if line % 2 == 0:  # Cr line: emit provisional (Cb may not be known yet)
            cb_src = line + 1 if self._have_cb[line + 1] else \
                     (line - 1 if line > 0 and self._have_cb[line - 1] else None)
            row = self._rgb_row(line, cb_src, line)
            events.append(SSTVEvent(EventType.LINE, mode=ROBOT_36, line=line, row=row))
        else:              # Cb line: finalize previous Cr line and this one
            if line > 0:
                row_prev = self._rgb_row(line - 1, line, line - 1)
                events.append(SSTVEvent(EventType.LINE, mode=ROBOT_36,
                                        line=line - 1, row=row_prev))
            row = self._rgb_row(line, line, line - 1 if line > 0 else None)
            events.append(SSTVEvent(EventType.LINE, mode=ROBOT_36, line=line, row=row))

        self.line = line + 1
        if self.line >= HEIGHT:
            self._finish(events, truncated=False)
            return events

        # schedule the next sync search
        nxt = sync + int(round(LINE_MS * _MS * fs))
        self._search_from = max(nxt - int(round(SYNC_SEARCH_EARLY_MS * _MS * fs)),
                                sync + int(round(60.0 * _MS * fs)))
        self._search_deadline = nxt + int(round(GIVE_UP_MS * _MS * fs))
        # keep a bounded amount of history
        self._trim(self._search_from - int(20 * _MS * fs))
        return events

    def _rgb_row(self, line: int, cb_line: Optional[int],
                 cr_line: Optional[int] = None) -> np.ndarray:
        y = self._y[line]
        cr = self._cr[cr_line if cr_line is not None else line]
        cb = self._cb[cb_line] if cb_line is not None else np.full(WIDTH, 128.0)
        return ycbcr_to_rgb(y, cb, cr)

    # ------------------------------------------------------------------
    def flush(self) -> List[SSTVEvent]:
        """End of stream: finish with whatever has been received."""
        events: List[SSTVEvent] = []
        if not self.done:
            self._finish(events, truncated=self.line < HEIGHT)
        return events

    def _finish(self, events: List[SSTVEvent], truncated: bool) -> None:
        self.done = True
        self.truncated = truncated
        quality = (self._syncs_found / self._expected_syncs
                   if self._expected_syncs else 0.0)
        # Robot 36 interleaves chroma: even lines carry Cr, odd lines Cb.
        # Assemble full-resolution Cb/Cr planes by pairing partner lines.
        cb_full = np.full((HEIGHT, WIDTH), 128.0)
        cr_full = np.full((HEIGHT, WIDTH), 128.0)
        for ln in range(HEIGHT):
            if ln % 2 == 0:  # Cr line
                if self._have_cr[ln]:
                    cr_full[ln] = self._cr[ln]
                src = ln + 1 if self._have_cb[ln + 1] else \
                    (ln - 1 if ln > 0 and self._have_cb[ln - 1] else None)
                if src is not None:
                    cb_full[ln] = self._cb[src]
            else:            # Cb line
                if self._have_cb[ln]:
                    cb_full[ln] = self._cb[ln]
                src = ln - 1 if ln > 0 and self._have_cr[ln - 1] else \
                    (ln + 1 if ln + 1 < HEIGHT and self._have_cr[ln + 1] else None)
                if src is not None:
                    cr_full[ln] = self._cr[src]
        image = ycbcr_to_rgb(self._y, cb_full, cr_full)
        # hide rows that never arrived, so truncation is visually obvious
        for ln in range(HEIGHT):
            if not self._have_y[ln]:
                image[ln] = 0
        events.append(SSTVEvent(EventType.IMAGE_COMPLETE, mode=ROBOT_36,
                                image=image, truncated=truncated,
                                sync_quality=quality))
        log.info("Robot36 image complete: %d lines, truncated=%s, sync=%.2f",
                 int(self._have_y.sum()), truncated, quality)


ROBOT_36 = SSTVMode(name="Robot 36", vis_code=8, width=WIDTH, height=HEIGHT,
                    duration_s=LINE_MS * HEIGHT / 1000.0,
                    decoder_factory=Robot36Decoder)
register_mode(ROBOT_36)

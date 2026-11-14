"""ISS pass simulator.

Generates a realistic pass without any hardware: SSTV test images are
FM-modulated onto a carrier that carries the *predicted doppler* from the
real TLE, plus optional noise.  The output is complex IQ exactly like an
RTL-SDR would deliver it, so the entire downstream chain (doppler
correction, FM demodulation, detection, decoding) is exercised for real.

A ``speed`` factor accelerates time for automated tests.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional

import numpy as np

from core.doppler import DopplerModel
from core.dsp import FMModulator, Preemphasis
from core.sources.base import Chunk, TimedSource
from core.tracker import PassInfo, Tracker
from sstv.encoder import encode_robot36, test_image

log = logging.getLogger(__name__)


class SimulatorSource(TimedSource):
    """Simulates an ISS SSTV pass as complex IQ."""

    def __init__(self,
                 tracker: Tracker,
                 pass_info: PassInfo,
                 freq_hz: float = 145_800_000.0,
                 tuning_offset_hz: float = 20_000.0,
                 deviation_hz: float = 3500.0,
                 images: Optional[List[np.ndarray]] = None,
                 snr_db: float = 15.0,
                 speed: float = 1.0,
                 fs_iq: float = 240_000.0,
                 chunk_samples: int = 24_000,
                 preroll_s: float = 10.0,
                 postroll_s: float = 10.0,
                 image_gap_s: float = 8.0,
                 first_image_delay_s: float = 20.0,
                 label: str = "simulator"):
        super().__init__(fs_iq, chunk_samples, is_iq=True, label=label)
        self.tracker = tracker
        self.pass_info = pass_info
        self.freq_hz = float(freq_hz)
        self.tuning_offset_hz = float(tuning_offset_hz)
        self.deviation_hz = float(deviation_hz)
        self.snr_db = float(snr_db)
        self.speed = float(speed)
        self.preroll_s = float(preroll_s)
        self.postroll_s = float(postroll_s)
        self.image_gap_s = float(image_gap_s)
        self.first_image_delay_s = float(first_image_delay_s)
        self._doppler = DopplerModel(tracker, self.freq_hz)
        self._mod = FMModulator(fs_iq, deviation_hz)
        self._pre = Preemphasis(fs_iq)
        self._rng = np.random.default_rng()
        # image audio at fs_iq: schedule (start_s, audio array) over the pass
        self._images = images if images is not None else [test_image(), test_image()]
        self._schedule: List[tuple] = []
        self._build_schedule()

    # ------------------------------------------------------------------
    def _build_schedule(self) -> None:
        dur = self.pass_info.duration_s
        if not self._images:
            self._schedule = []
            return
        img_dur = 0.910 + 36.0  # header + image
        starts = []
        t = self.first_image_delay_s
        while t + img_dur + self.postroll_s < dur:
            starts.append(t)
            t += img_dur + self.image_gap_s
        if not starts and dur > img_dur:
            starts = [max(0.0, (dur - img_dur) / 2)]
        for i, start in enumerate(starts):
            img = self._images[i % len(self._images)]
            audio = encode_robot36(img, self.fs, amplitude=0.8).astype(np.float64)
            audio = self._pre.process(audio)
            audio = audio / max(np.max(np.abs(audio)), 1e-9)
            self._schedule.append((start, audio))
        log.info("Simulator: pass %s -> %s, %d image(s) scheduled",
                 self.pass_info.aos.strftime("%H:%M:%S"),
                 self.pass_info.los.strftime("%H:%M:%S"), len(self._schedule))

    # ------------------------------------------------------------------
    @property
    def sim_start(self) -> float:
        return self.pass_info.aos.timestamp() - self.preroll_s

    @property
    def sim_end(self) -> float:
        return self.pass_info.los.timestamp() + self.postroll_s

    def sim_time(self, wall_t: float) -> float:
        """Map a wall-clock time to simulated pass time."""
        if self._t0 is None:
            return self.sim_start
        return self.sim_start + (wall_t - self._t0) * self.speed

    # ------------------------------------------------------------------
    def _audio_for(self, t_start: float, n: int) -> np.ndarray:
        """Audio (pre-emphasized, unit peak) for the sim interval."""
        rel = t_start - self.pass_info.aos.timestamp()
        audio = np.zeros(n)
        if rel < 0 or rel > self.pass_info.duration_s:
            return audio  # outside the pass: no transmitter
        for start, clip in self._schedule:
            i0 = int(round((start - rel) * self.fs))
            if i0 >= n or i0 + len(clip) <= 0:
                continue
            a = max(i0, 0)
            b = min(i0 + len(clip), n)
            audio[a:b] = clip[a - i0:b - i0]
        return audio

    def _loop(self, sink: Callable[[Chunk], None]) -> None:
        self._t0 = time.time()
        sim_t = self.sim_start
        wall = self._t0
        n = self.chunk_samples
        chunk_dur = n / self.fs
        noise_sigma = 10.0 ** (-self.snr_db / 20.0) / np.sqrt(2.0)
        while sim_t < self.sim_end and not self._stop_evt.is_set():
            audio = self._audio_for(sim_t, n)
            t_end = sim_t + (n - 1) / self.fs
            dop = self._doppler.segment(sim_t, t_end, n)
            offsets = self.tuning_offset_hz + dop
            iq = self._mod.process(audio, offsets)
            if noise_sigma > 0:
                iq = iq + noise_sigma * (self._rng.normal(size=n)
                                         + 1j * self._rng.normal(size=n)).astype(np.complex64)
                iq = iq.astype(np.complex64)
            sink(Chunk(data=iq, t=wall, is_iq=True, fs=self.fs))
            self._samples += n
            sim_t += chunk_dur
            wall = self._t0 + (sim_t - self.sim_start) / self.speed
            self._pace(wall, self.speed)

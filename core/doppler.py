"""Doppler model: converts satellite range-rate into a tunable frequency profile.

The SDR is parked at a fixed center frequency; the DSP chain removes doppler
in software.  This class provides thread-safe, cheap, interpolated doppler
lookups for the DSP thread (and anyone else who asks).
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from core.tracker import C_KM_S, Tracker


class DopplerModel:
    """Interpolated doppler-shift lookup at a given nominal frequency.

    A coarse grid is computed in the background of whatever thread asks for
    coverage; lookups inside the covered span are pure numpy interpolation.
    """

    GRID_STEP_S = 5.0
    PAD_S = 30.0

    def __init__(self, tracker: Tracker, freq_hz: float, enabled: bool = True):
        self.tracker = tracker
        self.freq_hz = float(freq_hz)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._t0 = 0.0  # unix seconds of first grid point
        self._dt = self.GRID_STEP_S
        self._grid = np.zeros(0)  # doppler Hz

    # ------------------------------------------------------------------
    def _ensure_cover(self, t_lo: float, t_hi: float) -> None:
        """(Re)build the cached grid so [t_lo, t_hi] is covered."""
        t_lo -= self.PAD_S
        t_hi += self.PAD_S
        with self._lock:
            n = self._grid.size
            if n and self._t0 <= t_lo and (self._t0 + (n - 1) * self._dt) >= t_hi:
                return
        # build fresh grid (outside lock: skyfield call)
        import datetime as _dt
        count = int(np.ceil((t_hi - t_lo) / self._dt)) + 1
        times = [_dt.datetime.fromtimestamp(t_lo + i * self._dt, _dt.timezone.utc)
                 for i in range(count)]
        if self.enabled:
            poss = self.tracker.positions(times)
            dop = np.array([-p.range_rate_km_s / C_KM_S * self.freq_hz for p in poss])
        else:
            dop = np.zeros(count)
        with self._lock:
            self._t0 = t_lo
            self._grid = dop

    # ------------------------------------------------------------------
    def doppler_at(self, t_unix: float) -> float:
        """Doppler at a single unix timestamp (Hz)."""
        self._ensure_cover(t_unix - 1.0, t_unix + 1.0)
        with self._lock:
            return float(np.interp(t_unix, self._grid_t(), self._grid))

    def segment(self, t_start: float, t_end: float, n: int) -> np.ndarray:
        """n doppler samples, linearly interpolated over [t_start, t_end]."""
        if n <= 0:
            return np.zeros(0)
        self._ensure_cover(min(t_start, t_end), max(t_start, t_end))
        ts = np.linspace(t_start, t_end, n)
        with self._lock:
            grid_t = self._grid_t()
            if self._grid.size < 2:
                return np.zeros(n)
            return np.interp(ts, grid_t, self._grid)

    def _grid_t(self) -> np.ndarray:
        if self._grid.size == 0:
            return np.zeros(0)
        return self._t0 + np.arange(self._grid.size) * self._dt

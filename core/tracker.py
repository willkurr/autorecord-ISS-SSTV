"""Satellite pass prediction and live tracking (SGP4 via Skyfield)."""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

from core.tle import TLE

log = logging.getLogger(__name__)

C_KM_S = 299792.458  # speed of light, km/s


@dataclass(frozen=True)
class PassInfo:
    """A single satellite pass over the ground station."""
    aos: datetime          # acquisition of signal (UTC, 0 deg elevation)
    los: datetime          # loss of signal (UTC)
    max_el_deg: float
    aos_az_deg: float
    los_az_deg: float

    @property
    def duration_s(self) -> float:
        return (self.los - self.aos).total_seconds()

    @property
    def in_progress_at(self) -> None:
        return None

    def contains(self, when: datetime) -> bool:
        return self.aos <= when <= self.los


@dataclass(frozen=True)
class SkyPosition:
    az_deg: float
    el_deg: float
    range_km: float
    range_rate_km_s: float  # positive = receding

    def doppler_hz(self, freq_hz: float) -> float:
        """Doppler shift of a transmitted carrier (positive = received high)."""
        return -self.range_rate_km_s / C_KM_S * freq_hz


class Tracker:
    """Track one satellite from a TLE for one ground station."""

    def __init__(self, tle: TLE):
        self.ts = load.timescale()
        self.tle = tle
        self.sat = EarthSatellite(tle.line1, tle.line2, tle.name, self.ts)
        self.station = None
        self._station_latlon = None

    # ------------------------------------------------------------- station
    def set_station(self, lat_deg: float, lon_deg: float, elevation_m: float = 0.0) -> None:
        self.station = wgs84.latlon(latitude_degrees=lat_deg,
                                     longitude_degrees=lon_deg,
                                     elevation_m=elevation_m)
        self._station_latlon = (lat_deg, lon_deg, elevation_m)

    def _require_station(self) -> None:
        if self.station is None:
            raise RuntimeError("Ground station coordinates not set (set_station)")

    # ------------------------------------------------------------- positions
    def _times(self, whens: Sequence[datetime]):
        dts = [w if w.tzinfo else w.replace(tzinfo=timezone.utc) for w in whens]
        return self.ts.from_datetimes(dts)

    def positions(self, whens: Sequence[datetime]) -> List[SkyPosition]:
        """Vectorized topocentric positions + radial velocities."""
        self._require_station()
        t = self._times(whens)
        topo = (self.sat - self.station).at(t)
        pos = topo.position.km
        if pos.ndim == 1:
            pos = pos[:, None]
        rng = np.linalg.norm(pos, axis=0)
        aa = topo.altaz()
        az_deg = np.atleast_1d(aa[1].degrees)
        el_deg = np.atleast_1d(aa[0].degrees)
        rr = self._range_rates(t, pos, rng)
        return [SkyPosition(float(a), float(e), float(r), float(v))
                for a, e, r, v in zip(az_deg, el_deg, rng, rr)]

    def _range_rates(self, t, pos, rng) -> np.ndarray:
        try:
            vel = (self.sat - self.station).at(t).velocity.km_per_s
            if vel is None:
                raise AttributeError
            if vel.ndim == 1:
                vel = vel[:, None]
            return np.einsum("ij,ij->j", pos, vel) / rng
        except Exception:  # noqa: BLE001 - numeric fallback
            log.debug("velocity unavailable; using numeric range-rate")
            rr = np.zeros(pos.shape[1])
            dts = np.atleast_1d(t.utc_datetime())
            for i in range(pos.shape[1]):
                rr[i] = self._numeric_range_rate(dts[i])
            return rr

    def _numeric_range_rate(self, t, dt_s: float = 0.5) -> float:
        t0, t1 = t.utc_datetime() - timedelta(seconds=dt_s), t.utc_datetime() + timedelta(seconds=dt_s)
        p0 = (self.sat - self.station).at(self.ts.from_datetimes([t0])).position.km
        p1 = (self.sat - self.station).at(self.ts.from_datetimes([t1])).position.km
        return (np.linalg.norm(p1) - np.linalg.norm(p0)) / (2 * dt_s)

    def position_at(self, when: datetime) -> SkyPosition:
        return self.positions([when])[0]

    # ------------------------------------------------------------- passes
    def next_passes(self, t_start: datetime, hours: float = 96.0,
                    min_el_deg: float = 10.0, limit: int = 40) -> List[PassInfo]:
        """Predict passes within the next `hours`, with max elevation >= min_el_deg."""
        self._require_station()
        t_start = t_start if t_start.tzinfo else t_start.replace(tzinfo=timezone.utc)
        t_end = t_start + timedelta(hours=hours)
        t0 = self.ts.from_datetime(t_start)
        t1 = self.ts.from_datetime(t_end)

        passes = self._assemble_passes(t0, t1)
        result: List[PassInfo] = []
        now_pos = self.position_at(t_start)
        if now_pos.el_deg > 0.0:
            # Pass already in progress: back-search for the true AOS.
            aos = self._search_aos(t_start, max_back_s=20 * 60)
            passes.insert(0, (aos, t_start, True))
        result = []
        for aos_dt, los_dt, _ in passes:
            if los_dt <= t_start:
                continue
            info = self._make_pass_info(aos_dt, los_dt)
            if info.max_el_deg >= min_el_deg:
                result.append(info)
        # merge an in-progress pass if its AOS search overlapped a predicted one
        merged: List[PassInfo] = []
        for p in sorted(result, key=lambda p: p.aos):
            if merged and p.aos <= merged[-1].los:
                # overlapping duplicate (in-progress + predicted); keep higher max el
                if p.max_el_deg > merged[-1].max_el_deg:
                    merged[-1] = p
            else:
                merged.append(p)
        return merged[:limit]

    def _assemble_passes(self, t0, t1) -> List[Tuple[datetime, datetime, bool]]:
        events = self.sat.find_events(self.station, t0, t1, altitude_degrees=0.0)
        out: List[Tuple[datetime, datetime, bool]] = []
        aos: Optional[datetime] = None
        for t_ev, ev in zip(*events):
            dt = t_ev.utc_datetime()
            if ev == 0:  # rise
                aos = dt
            elif ev == 2 and aos is not None:  # set
                out.append((aos, dt, True))
                aos = None
        return out

    def _search_aos(self, t_now: datetime, max_back_s: float) -> datetime:
        step = timedelta(seconds=30)
        t = t_now
        for _ in range(int(max_back_s / 30) + 2):
            prev = t - step
            if self.position_at(prev).el_deg <= 0.0:
                return prev  # coarse is good enough for an in-progress pass
            t = prev
        return t_now

    def _make_pass_info(self, aos: datetime, los: datetime) -> PassInfo:
        span_s = (los - aos).total_seconds()
        n = 60
        whens = [aos + timedelta(seconds=span_s * (i + 0.5) / n) for i in range(n)]
        poss = self.positions(whens)
        max_el = max(p.el_deg for p in poss)
        return PassInfo(aos=aos, los=los, max_el_deg=max_el,
                        aos_az_deg=poss[0].az_deg, los_az_deg=poss[-1].az_deg)

    def pass_path(self, info: PassInfo, step_s: float = 10.0
                  ) -> Tuple[List[float], List[float]]:
        """Sampled az/el across a pass, for drawing the track on the sky map."""
        span = info.duration_s
        times = [info.aos + timedelta(seconds=s)
                 for s in np.arange(0, span + step_s / 2, step_s)]
        poss = self.positions(times)
        return ([p.az_deg for p in poss], [p.el_deg for p in poss])

    def current_pass(self, when: datetime, lookahead_s: float = 0.0) -> Optional[PassInfo]:
        """The pass covering `when`, if the satellite is up (or about to be)."""
        if self.position_at(when).el_deg > 0:
            aos = self._search_aos(when, max_back_s=20 * 60)
            los = self._find_los(aos)
            return self._make_pass_info(aos, los)
        return None

    def _find_los(self, aos: datetime, max_forward_s: float = 20 * 60) -> datetime:
        step = timedelta(seconds=30)
        t = aos
        for _ in range(int(max_forward_s / 30) + 2):
            t += step
            if self.position_at(t).el_deg <= 0.0:
                return t
        return aos + timedelta(minutes=15)

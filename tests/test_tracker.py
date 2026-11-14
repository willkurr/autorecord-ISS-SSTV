from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.doppler import DopplerModel
from core.tracker import C_KM_S, PassInfo, Tracker


def test_position_sane(tracker):
    now = datetime.now(timezone.utc)
    pos = tracker.position_at(now)
    assert -90.0 <= pos.el_deg <= 90.0
    assert 0.0 <= pos.az_deg < 360.0
    assert 300.0 < pos.range_km < 25000.0
    assert -8.0 < pos.range_rate_km_s < 8.0


def test_range_rate_matches_numeric(tracker):
    now = datetime.now(timezone.utc)
    pos = tracker.position_at(now)
    t = tracker.ts.from_datetime(now)
    num = tracker._numeric_range_rate(t)
    assert abs(pos.range_rate_km_s - num) < 0.05  # km/s agreement


def test_pass_prediction_structure(tracker):
    t0 = datetime.now(timezone.utc)
    passes = tracker.next_passes(t0, hours=48.0, min_el_deg=5.0)
    assert len(passes) >= 2  # ISS passes ~5-8x/day above 5 deg from mid-latitudes
    for p in passes:
        assert p.aos < p.los
        assert 60.0 < p.duration_s < 15 * 60.0
        assert 5.0 <= p.max_el_deg <= 90.0
        assert 0.0 <= p.aos_az_deg < 360.0
        assert 0.0 <= p.los_az_deg < 360.0
        assert p.los > t0
    # sorted by AOS, no overlaps
    for a, b in zip(passes, passes[1:]):
        assert a.aos < b.aos
        assert a.los <= b.aos + timedelta(seconds=1)


def test_pass_path(tracker):
    t0 = datetime.now(timezone.utc)
    passes = tracker.next_passes(t0, hours=24.0, min_el_deg=20.0)
    if not passes:
        pytest.skip("no high pass in window")
    azs, els = tracker.pass_path(passes[0], step_s=15.0)
    assert len(azs) == len(els)
    assert max(els) > 15.0
    assert all(0 <= a < 360 for a in azs)


def test_doppler_model(tracker):
    now = datetime.now(timezone.utc)
    freq = 145_800_000.0
    model = DopplerModel(tracker, freq)
    t0 = now.timestamp()
    seg = model.segment(t0, t0 + 120.0, 25)
    assert seg.shape == (25,)
    # ISS doppler at 145.8 MHz stays within ~+-4 kHz
    assert np.all(np.abs(seg) <= 4000.0)
    # doppler_at agrees with the segment endpoints
    assert abs(model.doppler_at(t0) - seg[0]) < 1.0
    assert abs(model.doppler_at(t0 + 120.0) - seg[-1]) < 1.0


def test_doppler_matches_direct_computation(tracker):
    from datetime import datetime as dte, timezone as tz
    freq = 145_800_000.0
    when = dte.now(tz.utc)
    model = DopplerModel(tracker, freq)
    via_model = model.doppler_at(when.timestamp())
    pos = tracker.position_at(when)
    direct = pos.doppler_hz(freq)
    assert abs(via_model - direct) < 20.0  # Hz; grid interp tolerance

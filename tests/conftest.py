"""Shared fixtures for the test suite."""
from __future__ import annotations

import pytest

from core.tle import TLE

# Snapshot of the AMSAT daily feed (epoch 2026-11-12) used for offline tests.
AMSAT_SNIPPET = """AO-07
1 07530U 74089B   26316.65907659 -.00000043  00000-0  22025-4 0  9991
2 07530 101.9874 331.4390 0012135 207.0986 271.2147 12.53700555379123
ISS
1 25544U 98067A   26316.43260916  .00008191  00000-0  15481-3 0  9990
2 25544  51.6324 286.4897 0007327  17.4271 342.6968 15.49332729594739
SO-50
1 27607U 02058C   26316.49286662  .00001582  00000-0  20311-3 0  9999
2 27607  64.5539  30.0389 0068124 235.0518 124.8378 14.83333417286454
"""

ISS_TLE = TLE(
    name="ISS",
    line1="1 25544U 98067A   26316.43260916  .00008191  00000-0  15481-3 0  9990",
    line2="2 25544  51.6324 286.4897 0007327  17.4271 342.6968 15.49332729594739",
)


@pytest.fixture
def iss_tle() -> TLE:
    return ISS_TLE


@pytest.fixture
def tracker(iss_tle):
    from core.tracker import Tracker
    tr = Tracker(ISS_TLE)
    tr.set_station(lat_deg=40.0, lon_deg=-105.0, elevation_m=1600.0)
    return tr


@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(20261112)

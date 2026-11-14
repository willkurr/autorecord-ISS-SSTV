"""End-to-end tests: simulator -> receive chain -> engine -> images."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.doppler import DopplerModel
from core.dsp import ReceiveChain
from core.recorder import PassRecorder
from core.sources.sim import SimulatorSource
from core.tracker import PassInfo, Tracker
from sstv.encoder import test_image
from sstv.engine import SSTVEngine, decode_file
from sstv.events import EventType

SPEED = 40.0  # time acceleration for the test


@pytest.fixture(scope="module")
def tca_pass():
    """A short synthetic window centred on a real pass culmination, so the
    doppler profile has realistic dynamics (fast zero-crossing at TCA)."""
    from tests.conftest import ISS_TLE
    tracker = Tracker(ISS_TLE)
    tracker.set_station(lat_deg=40.0, lon_deg=-105.0, elevation_m=1600.0)
    now = datetime.now(timezone.utc)
    passes = tracker.next_passes(now, hours=24.0, min_el_deg=20.0)
    if not passes:
        pytest.skip("no high pass within 24 h of the fixture epoch")
    p = passes[0]
    tca = p.aos + timedelta(seconds=p.duration_s / 2.0)
    return tracker, PassInfo(aos=tca - timedelta(seconds=60.0),
                             los=tca + timedelta(seconds=60.0),
                             max_el_deg=p.max_el_deg,
                             aos_az_deg=p.aos_az_deg, los_az_deg=p.los_az_deg)


def test_full_simulated_pass(tca_pass, tmp_path):
    """Simulator (real doppler + FM + noise) -> chain -> engine -> image."""
    tracker, pinfo = tca_pass
    freq = 145_800_000.0
    sim = SimulatorSource(
        tracker, pinfo, freq_hz=freq, images=[test_image()],
        snr_db=15.0, speed=SPEED, preroll_s=5.0, postroll_s=5.0,
        first_image_delay_s=20.0)

    dop_model = DopplerModel(tracker, freq)

    def chain_dop(t0, t1, n):
        # map wall-clock chunk times back into simulated pass time
        return dop_model.segment(sim.sim_time(t0), sim.sim_time(t1), n)

    chain = ReceiveChain(doppler_segment=chain_dop)
    events = []
    engine = SSTVEngine(48_000.0, on_event=events.append)

    def sink(chunk):
        audio = chain.process(chunk.data, chunk.t)
        engine.process(audio)

    sim.start(sink)
    deadline = time.time() + 240.0
    while sim.running and time.time() < deadline:
        time.sleep(0.2)
    sim.stop()
    engine.flush()

    starts = [e for e in events if e.type == EventType.IMAGE_START]
    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    # the 120 s window fits two scheduled images (20 s + 36.9 s + 8 s gap)
    assert len(starts) == 2, f"expected 2 image starts, got {len(starts)}"
    assert len(comps) == 2
    for comp in comps:
        assert not comp.truncated
        assert comp.sync_quality >= 0.98

    img = test_image()
    y_orig = img.mean(axis=2).ravel()
    for comp in comps:
        y_dec = comp.image.mean(axis=2).ravel()
        assert np.corrcoef(y_orig, y_dec)[0, 1] > 0.9
        assert np.abs(comp.image.astype(float) - img.astype(float)).mean() < 25.0


def test_noise_only_pass_no_false_detection(tca_pass):
    """A carrier-only noisy pass (no images) must not trigger detection."""
    tracker, pinfo = tca_pass
    freq = 145_800_000.0
    sim = SimulatorSource(
        tracker, pinfo, freq_hz=freq, images=[], snr_db=15.0, speed=SPEED,
        preroll_s=5.0, postroll_s=5.0)
    dop_model = DopplerModel(tracker, freq)

    def chain_dop(t0, t1, n):
        return dop_model.segment(sim.sim_time(t0), sim.sim_time(t1), n)

    chain = ReceiveChain(doppler_segment=chain_dop)
    events = []
    engine = SSTVEngine(48_000.0, on_event=events.append)

    def sink(chunk):
        engine.process(chain.process(chunk.data, chunk.t))

    sim.start(sink)
    deadline = time.time() + 240.0
    while sim.running and time.time() < deadline:
        time.sleep(0.2)
    sim.stop()
    engine.flush()

    starts = [e for e in events if e.type == EventType.IMAGE_START]
    assert starts == [], f"false detections: {len(starts)}"


def test_recorder_roundtrip(tmp_path):
    """Recorded pass audio re-decodes from the WAV file via decode_file."""
    from sstv.encoder import encode_robot36
    img = test_image()
    audio = encode_robot36(img, 48_000.0)
    rng = np.random.default_rng(11)
    stream = np.concatenate([
        rng.normal(0, 0.03, 48_000).astype(np.float32),
        audio,
        rng.normal(0, 0.03, 48_000).astype(np.float32),
    ])

    rec = PassRecorder(tmp_path, satellite="ISS", freq_hz=145_800_000.0,
                       meta={"tle": "test"})
    rec.start()
    for i in range(0, len(stream), 4800):
        rec.write(stream[i:i + 4800])
    rec.note_image("ignored.png", {"mode": "Robot 36"})
    wav_path = rec.close()
    assert wav_path is not None and wav_path.exists()
    assert rec.meta_path.exists()

    comps = decode_file(wav_path)
    assert len(comps) == 1
    assert not comps[0].truncated
    y_orig = img.mean(axis=2).ravel()
    y_dec = comps[0].image.mean(axis=2).ravel()
    assert np.corrcoef(y_orig, y_dec)[0, 1] > 0.95

"""Receive-chain tests: FM + doppler through the IQ -> audio pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from core.dsp import FMModulator, Preemphasis, ReceiveChain
from core.doppler import DopplerModel
from core.tracker import Tracker
from sstv.encoder import encode_robot36, test_image
from sstv.engine import SSTVEngine
from sstv.events import EventType
from sstv.freq import StreamingAnalytic
from tests.conftest import ISS_TLE

FS_IQ = 240_000
FS_AUDIO = 48_000
T0 = 1_700_000_000.0  # arbitrary unix epoch for the test
OFFSET = 20_000.0
DEV = 3500.0


def doppler_fn(t):
    """Realistic ISS doppler profile at 145.8 MHz (+-3.6 kHz, ~75 Hz/s)."""
    return 3600.0 * np.sin(2.0 * np.pi * (np.asarray(t, dtype=float) - T0) / 300.0)


def make_iq(audio, t_start, doppler, offset=OFFSET, dev=DEV, noise_sigma=0.0):
    """Non-streaming test modulator: audio at fs_iq -> IQ at fs_iq."""
    n = len(audio)
    ts = t_start + np.arange(n) / FS_IQ
    offsets = offset + doppler(ts)
    mod = FMModulator(FS_IQ, dev)
    iq = mod.process(audio, offsets)
    if noise_sigma > 0:
        iq = iq + (noise_sigma / np.sqrt(2)) * (
            np.random.default_rng(3).normal(size=n)
            + 1j * np.random.default_rng(4).normal(size=n)).astype(np.complex128)
    return iq.astype(np.complex64)


def tone(freq, seconds, fs=FS_IQ, amp=0.8):
    n = int(seconds * fs)
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / fs).astype(np.float64)


def test_tone_survives_doppler():
    """A 1900 Hz FM tone with a doppler ramp demodulates back to 1900 Hz."""
    audio = tone(1900.0, 2.0)
    iq = make_iq(audio, T0, doppler_fn)
    chain = ReceiveChain(doppler_segment=lambda t0, t1, n: doppler_fn(
        np.linspace(t0, t1, n)), fine_offset_hz=0.0)
    out = []
    chunk = 24_000
    for i in range(0, len(iq), chunk):
        t = T0 + i / FS_IQ
        out.append(chain.process(iq[i:i + chunk], t))
    out = np.concatenate(out)
    # analyse the recovered tone (skip the first 200 ms of transients)
    sa = StreamingAnalytic(FS_AUDIO)
    frame = sa.process(out)
    mid = frame.freq[int(0.2 * FS_AUDIO):int(1.8 * FS_AUDIO)]
    assert abs(np.median(mid) - 1900.0) < 25.0
    assert np.percentile(np.abs(mid - 1900.0), 90) < 80.0


def test_dsp_keeps_up_in_real_time():
    """The full chain+engine must process a 50 ms chunk well under 50 ms.

    Regression guard: the detector once scanned its whole rolling buffer
    in a Python loop every chunk (~10 ms here, far worse on other
    machines), starving the live monitor.  Noise input keeps the
    detector in its seeking state, the expensive path.
    """
    import time as _time
    tracker = Tracker(ISS_TLE)
    tracker.set_station(40.0, -105.0, 1600.0)
    dop = DopplerModel(tracker, 145_800_000.0)
    chain = ReceiveChain(
        doppler_segment=lambda t0, t1, n: dop.segment(t0, t1, n))
    engine = SSTVEngine(48_000.0)
    rng = np.random.default_rng(1)
    iq = (rng.normal(size=12_000)
          + 1j * rng.normal(size=12_000)).astype(np.complex64)
    chain.process(iq, 1_700_000_000.0)   # warm-up (allocations, caches)
    engine.process(np.zeros(2400, np.float32))

    n = 60
    t0 = _time.perf_counter()
    for k in range(n):
        t = 1_700_000_000.0 + k * 12_000 / 240_000.0
        engine.process(chain.process(iq, t))
    per_chunk_ms = (_time.perf_counter() - t0) / n * 1000.0
    assert per_chunk_ms < 25.0, (
        f"DSP takes {per_chunk_ms:.1f} ms per 50 ms chunk - the live "
        "monitor will starve")
    assert engine.detector._env_ref >= 0  # seek path was exercised


def test_channel_passes_full_carson_bandwidth():
    """The 15 kHz channel must pass the full Carson bandwidth of the FM
    signal (2*(deviation + max_tone) = 11.6 kHz) without distortion."""
    audio = tone(2300.0, 2.0)  # the highest SSTV tone (white)
    iq = make_iq(audio, T0, doppler_fn, dev=3500.0)
    chain = ReceiveChain(
        doppler_segment=lambda t0, t1, n: doppler_fn(np.linspace(t0, t1, n)))
    out = []
    for i in range(0, len(iq), 24_000):
        out.append(chain.process(iq[i:i + 24_000], T0 + i / FS_IQ))
    out = np.concatenate(out)
    sa = StreamingAnalytic(FS_AUDIO)
    frame = sa.process(out)
    mid = frame.freq[int(0.2 * FS_AUDIO):int(1.8 * FS_AUDIO)]
    assert abs(np.median(mid) - 2300.0) < 30.0
    assert np.percentile(np.abs(mid - 2300.0), 95) < 90.0


def test_fine_offset_correction():
    """A residual +300 Hz error is removed via fine_offset_hz."""
    audio = tone(1500.0, 1.5)
    iq = make_iq(audio, T0, doppler_fn, offset=OFFSET + 300.0)
    chain = ReceiveChain(
        doppler_segment=lambda t0, t1, n: doppler_fn(np.linspace(t0, t1, n)),
        fine_offset_hz=300.0)
    out = []
    chunk = 24_000
    for i in range(0, len(iq), chunk):
        out.append(chain.process(iq[i:i + chunk], T0 + i / FS_IQ))
    out = np.concatenate(out)
    sa = StreamingAnalytic(FS_AUDIO)
    frame = sa.process(out)
    mid = frame.freq[int(0.2 * FS_AUDIO):int(1.3 * FS_AUDIO)]
    assert abs(np.median(mid) - 1500.0) < 25.0


def test_sstv_image_through_rf_chain():
    """End-to-end: encoded SSTV -> preemphasis -> FM+doppler+noise -> chain
    -> audio -> SSTV engine -> decoded image."""
    img = test_image()
    # encode directly at the IQ rate (no resampling needed)
    audio = encode_robot36(img, FS_IQ, amplitude=0.8).astype(np.float64)
    pre = Preemphasis(FS_IQ)
    audio_pe = pre.process(audio)
    # normalize so the FM deviation peaks near DEV
    audio_pe = audio_pe / np.max(np.abs(audio_pe)) * (DEV / 3500.0)

    # 3.6 kHz doppler + noise at ~13 dB SNR
    iq = make_iq(audio_pe, T0, doppler_fn, noise_sigma=0.22)

    chain = ReceiveChain(
        doppler_segment=lambda t0, t1, n: doppler_fn(np.linspace(t0, t1, n)))
    events = []
    eng = SSTVEngine(FS_AUDIO, on_event=events.append)
    chunk = 24_000
    for i in range(0, len(iq), chunk):
        t = T0 + i / FS_IQ
        audio_out = chain.process(iq[i:i + chunk], t)
        eng.process(audio_out)
    eng.flush()

    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    assert len(comps) == 1
    comp = comps[0]
    assert not comp.truncated
    assert comp.sync_quality >= 0.99
    y_orig = img.mean(axis=2)
    y_dec = comp.image.mean(axis=2)
    err = np.abs(y_dec - y_orig)
    assert err.mean() < 12.0
    assert np.corrcoef(y_orig.ravel(), y_dec.ravel())[0, 1] > 0.93

"""Round-trip tests: encode Robot 36 audio, decode it, compare images."""
from __future__ import annotations

import numpy as np
import pytest

from sstv.encoder import VIS_CODE, encode_robot36, test_image
from sstv.engine import SSTVEngine
from sstv.events import EventType
from sstv.freq import hilbert_fir_taps
from sstv.modes import mode_for_vis

FS = 48000


def feed(engine: SSTVEngine, audio: np.ndarray, rng, max_chunk=9600):
    """Feed audio in random-sized chunks (exercises streaming state)."""
    i = 0
    while i < len(audio):
        n = int(rng.integers(400, max_chunk))
        engine.process(audio[i:i + n])
        i += n


def collect_events(engine: SSTVEngine, audio, rng):
    events = []
    eng = SSTVEngine(FS, on_event=events.append)
    feed(eng, audio, rng)
    eng.flush()
    return events


def test_hilbert_taps_passband():
    """The FIR Hilbert must be flat across SSTV tones (1100-2300 Hz)."""
    taps = hilbert_fir_taps(513)
    from numpy.fft import rfft
    h = np.abs(rfft(taps, n=32768))
    freqs = np.fft.rfftfreq(32768, d=1.0 / FS)
    for f in (1100.0, 1200.0, 1500.0, 1900.0, 2300.0):
        m = h[np.searchsorted(freqs, f)]
        assert abs(m - 1.0) < 0.05, f"Hilbert response off at {f} Hz: {m}"


def test_mode_registry():
    m = mode_for_vis(VIS_CODE)
    assert m is not None and m.name == "Robot 36"
    assert (m.width, m.height) == (320, 240)


def test_roundtrip_clean(rng):
    img = test_image()
    audio = encode_robot36(img, FS)
    events = collect_events(None, audio, rng)

    starts = [e for e in events if e.type == EventType.IMAGE_START]
    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    assert len(starts) == 1
    assert len(comps) == 1
    comp = comps[0]
    assert not comp.truncated
    assert comp.sync_quality >= 0.99
    assert comp.image.shape == (240, 320, 3)

    err = np.abs(comp.image.astype(int) - img.astype(int))
    # colour-bar interiors must survive; hard tone transitions smear a
    # couple of pixels (finite bandwidth), so the mean includes that smear
    assert err.mean() < 8.0, f"mean abs error {err.mean():.2f}"
    assert np.percentile(err, 90) < 10.0
    # Y channel fidelity on bar interiors (3 px away from boundaries and
    # from the start/end of each scan, where filter ramps smear)
    y_orig = img.mean(axis=2)
    y_dec = comp.image.mean(axis=2)
    interior = np.ones(320, dtype=bool)
    interior[:4] = False
    interior[-4:] = False
    for b in range(40, 320, 40):
        interior[max(0, b - 3):b + 3] = False
    assert np.abs(y_dec[:, interior] - y_orig[:, interior]).mean() < 3.0
    assert np.corrcoef(y_orig[:, interior].ravel(),
                       y_dec[:, interior].ravel())[0, 1] > 0.995


def test_roundtrip_with_silence_and_noise(rng):
    img = test_image()
    audio = encode_robot36(img, FS)
    noise = rng.normal(0, 0.05, FS).astype(np.float32)
    lead = np.concatenate([np.zeros(FS, np.float32), noise])
    trail = rng.normal(0, 0.05, FS).astype(np.float32)
    stream = np.concatenate([lead, audio, trail])
    events = collect_events(None, stream, rng)
    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    assert len(comps) == 1
    assert not comps[0].truncated
    err = np.abs(comps[0].image.astype(int) - img.astype(int))
    assert err.mean() < 10.0


def test_roundtrip_noisy_signal(rng):
    """Decode with heavy additive noise (approx 6 dB SNR on the tone)."""
    img = test_image()
    audio = encode_robot36(img, FS, amplitude=0.85)
    noisy = audio + rng.normal(0, 0.45, audio.shape).astype(np.float32)
    events = collect_events(None, noisy, rng)
    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    assert len(comps) == 1
    assert not comps[0].truncated
    # structure should still be visible
    y_orig = img.mean(axis=2).ravel()
    y_dec = comps[0].image.mean(axis=2).ravel()
    assert np.corrcoef(y_orig, y_dec)[0, 1] > 0.85


def test_two_images_back_to_back(rng):
    img = test_image()
    audio = np.concatenate([
        encode_robot36(img, FS),
        np.zeros(int(1.0 * FS), np.float32),
        encode_robot36(img, FS),
        np.zeros(int(0.5 * FS), np.float32),
    ])
    events = collect_events(None, audio, rng)
    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    assert len(comps) == 2
    assert all(not c.truncated for c in comps)


def test_truncated_transmission(rng):
    img = test_image()
    audio = encode_robot36(img, FS)
    cut = int((1.21 + 100 * 0.150) * FS)  # header + 100 lines
    events = collect_events(None, audio[:cut], rng)
    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    assert len(comps) == 1
    assert comps[0].truncated
    assert comps[0].image[:90].mean() > 8  # the received part is not all black


def test_bad_parity_rejected(rng):
    """Flip the VIS parity bit to the opposite valid tone -> no detection."""
    img = test_image()
    audio = encode_robot36(img, FS)
    # parity bit is the 8th bit after the start bit: sample range
    # [0.300+0.010+0.300+0.030 + 7*0.030, +0.030) s; the code has an odd
    # number of ones, so the encoder sends 1100 Hz - overwrite with 1300 Hz.
    i0 = int((0.300 + 0.010 + 0.300 + 0.030 + 7 * 0.030) * FS)
    i1 = i0 + int(0.030 * FS)
    # splice a 1300 Hz tone with phase matched at the segment start
    import math
    phase0 = 2 * math.pi * 1300.0 * i0 / FS
    t = np.arange(i1 - i0)
    audio[i0:i1] = 0.85 * np.sin(phase0 + 2 * np.pi * 1300.0 * t / FS)
    events = collect_events(None, audio, rng)
    starts = [e for e in events if e.type == EventType.IMAGE_START]
    assert starts == []


def test_streaming_chunk_sizes(rng):
    """Detector works with pathological chunking: 1-sample and huge chunks."""
    img = test_image()
    audio = encode_robot36(img, FS)
    events = []
    eng = SSTVEngine(FS, on_event=events.append)
    # a few tiny chunks first (breaks the header across many frames)
    eng.process(audio[:5])
    eng.process(audio[5:9])
    eng.process(audio[9:600])
    # then big ones
    i = 600
    while i < len(audio):
        eng.process(audio[i:i + 60000])
        i += 60000
    eng.flush()
    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    assert len(comps) == 1
    assert not comps[0].truncated

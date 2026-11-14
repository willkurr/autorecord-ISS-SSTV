"""Tests for the audio monitor reservoir (smoothness of the speaker feed)."""
from __future__ import annotations

import numpy as np
import pytest

from core.pipeline import AudioReservoir

FS = 48_000


def test_pull_before_prebuffer_returns_silence():
    r = AudioReservoir(FS, prebuffer_s=0.1)
    r.write(np.full(1000, 0.5, np.float32))  # well below the prebuffer
    assert np.all(r.pull(1024) == 0)
    # once the prebuffer level is reached, playback begins
    r.write(np.full(int(0.1 * FS), 0.5, np.float32))
    out = r.pull(1024)
    assert np.all(out == 0.5)


def test_playback_is_continuous_and_in_order():
    """Simulate bursty 100 ms chunk production vs small callback pulls.

    The concatenated playback must exactly equal the written stream - no
    gaps, no reordered remainders (the old queue-based monitor garbled
    this).
    """
    r = AudioReservoir(FS, prebuffer_s=0.05, max_buffer_s=2.0)
    rng = np.random.default_rng(5)
    written = []
    pulled = []
    chunk = int(0.1 * FS)  # 4800 samples per burst
    pull_sizes = [1024, 1024, 1024, 1024, 704]  # sums to exactly 4800
    for k in range(10):
        block = (rng.normal(0, 0.2, chunk) + k).astype(np.float32)
        r.write(block)
        written.append(block)
        for frames in pull_sizes:
            pulled.append(r.pull(frames))
    stream = np.concatenate(pulled)
    reference = np.concatenate(written)
    # the prebuffer delay shifts playback; align and compare
    start = int(np.argmax(stream != 0))
    n = min(len(reference), len(stream) - start)
    assert n >= len(reference) - 8  # essentially everything was played
    assert r.underruns == 0  # balanced production/pull: no dropouts at all
    assert np.allclose(stream[start:start + n], reference[:n], atol=1e-6)


def test_underrun_returns_silence_and_reprimes():
    r = AudioReservoir(FS, prebuffer_s=0.05)
    r.write(np.full(int(0.1 * FS), 0.5, np.float32))
    assert r.pull(1024)[0] == 0.5  # primed, playing
    # drain everything
    while True:
        out = r.pull(1024)
        if np.all(out == 0):
            break
    n_underruns = r.underruns
    assert n_underruns >= 1
    # a dribble of samples is not enough to resume (re-priming)
    r.write(np.full(100, 0.5, np.float32))
    assert np.all(r.pull(1024) == 0)
    # once re-primed, playback resumes
    r.write(np.full(int(0.1 * FS), 0.5, np.float32))
    assert r.pull(1024)[0] == 0.5


def test_partial_underrun_keeps_playing():
    """A short hiccup costs one small gap, not a full re-prime silence."""
    r = AudioReservoir(FS, prebuffer_s=0.05)
    r.write(np.full(3000, 0.5, np.float32))
    assert r.pull(1024)[0] == 0.5          # primed and playing
    assert r.pull(1024)[0] == 0.5          # buf: 3000 -> 1976 -> 952
    out = r.pull(1024)                     # partial: 952 real + 72 zeros
    assert out[0] == 0.5 and out[-1] == 0
    assert r.underruns == 1
    # still primed: the next small write plays immediately
    r.write(np.full(512, 0.5, np.float32))
    assert r.pull(1024)[0] == 0.5


def test_latency_is_bounded():
    """A producer that runs far ahead drops the oldest audio, not the newest."""
    r = AudioReservoir(FS, prebuffer_s=0.05, max_buffer_s=0.2)
    for k in range(20):
        r.write(np.full(4800, k, np.float32))  # producer runs far ahead
        r.pull(1024)
    with r._lock:
        held = len(r._buf)
    assert held <= int(0.2 * FS) + 4800  # bounded

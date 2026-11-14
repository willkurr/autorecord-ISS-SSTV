"""Streaming tone analysis: analytic signal + instantaneous frequency.

The SSTV signal is an FM-subcarried audio tone whose frequency (1500 Hz =
black to 2300 Hz = white) encodes pixel brightness.  Decoding therefore
means estimating the *instantaneous frequency* of the audio stream, sample
by sample, in a way that survives arbitrary chunk boundaries.

``StreamingAnalytic`` builds the analytic signal z[n] = x[n] + j*H{x}[n]
using a windowed-FIR Hilbert transformer with carried filter state, so it
is exact and continuous across chunks.  Instantaneous frequency follows
from the phase difference of adjacent samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin2, lfilter


def hilbert_fir_taps(length: int = 513) -> np.ndarray:
    """Windowed FIR approximating the Hilbert transformer (Type III).

    The passband is flat well below 1 kHz at audio rates with 513 taps,
    which matters because VIS bits are sent at 1100/1300 Hz.
    """
    if length % 2 == 0:
        raise ValueError("FIR Hilbert needs an odd tap count")
    m = (length - 1) // 2
    n = np.arange(-m, m + 1, dtype=float)
    h = np.zeros(length)
    odd = (n % 2) != 0
    h[odd] = 2.0 / (np.pi * n[odd])
    h *= np.blackman(length)
    return h


class StreamingBandpass:
    """Streaming FIR band-pass, used to clean audio before tone analysis.

    SSTV tones live between 1100 and 2300 Hz; stripping everything else
    removes the bulk of wideband noise (which otherwise biases the
    instantaneous-frequency estimate badly) and out-of-band interference.
    """

    def __init__(self, fs: float, lo: float = 950.0, hi: float = 2450.0,
                 numtaps: int = 511):
        freqs = [0.0, lo - 250.0, lo, hi, hi + 250.0, fs / 2.0]
        gain = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        self.taps = firwin2(numtaps, freqs, gain, fs=fs)
        self._zi = np.zeros(numtaps - 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return x
        y, self._zi = lfilter(self.taps, 1.0, x, zi=self._zi)
        return y


@dataclass
class ToneFrame:
    """Instantaneous frequency/envelope for a chunk of the audio stream.

    ``freq[i]`` is the tone frequency (Hz) at global sample index
    ``start + i``; ``env[i]`` is the analytic-signal magnitude.
    """
    freq: np.ndarray
    env: np.ndarray
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.freq)

    def window_mean(self, i0: int, i1: int) -> float:
        """Mean frequency over the global index range [i0, i1)."""
        lo = max(i0, self.start)
        hi = min(i1, self.start + len(self.freq))
        if hi <= lo:
            return float("nan")
        return float(np.mean(self.freq[lo - self.start:hi - self.start]))


class StreamingAnalytic:
    """Continuous analytic signal and instantaneous frequency of a stream."""

    def __init__(self, fs: float, taps: np.ndarray | None = None):
        self.fs = float(fs)
        self.taps = taps if taps is not None else hilbert_fir_taps()
        self._half = (len(self.taps) - 1) // 2
        self._zi = np.zeros(len(self.taps) - 1)
        self._pending = np.zeros(self._half)  # real-sample delay line
        self._n_out = 0
        self._last_z: complex | None = None

    def process(self, x: np.ndarray) -> ToneFrame:
        x = np.asarray(x, dtype=np.float64).ravel()
        n = x.size
        if n == 0:
            return ToneFrame(np.zeros(0), np.zeros(0), self._n_out)

        # Imaginary branch: FIR Hilbert with carried state (delay = _half).
        y, self._zi = lfilter(self.taps, 1.0, x, zi=self._zi)
        # Real branch: the same delay, via a sample delay line.
        buf = np.concatenate([self._pending, x])
        real = buf[:n]
        self._pending = buf[n:]

        z = real + 1j * y
        if self._last_z is None:
            zz = np.concatenate([z[:1], z])
        else:
            zz = np.concatenate([[self._last_z], z])
        dz = zz[1:] * np.conj(zz[:-1])
        freq = np.angle(dz) * (self.fs / (2.0 * np.pi))
        env = np.abs(z)
        self._last_z = z[-1]

        start = self._n_out
        self._n_out += n
        return ToneFrame(freq, env, start)

    def flush(self) -> ToneFrame:
        """Drain the delay line: emit outputs for the final buffered samples.

        The real branch delays by ``_half`` samples, so after the last
        ``process()`` call that many true samples are still pending.  Feeding
        ``_half`` zeros through the same path releases them (the imaginary
        branch becomes approximate near the very end, which is harmless).
        """
        if self._pending.size == 0:
            return ToneFrame(np.zeros(0), np.zeros(0), self._n_out)
        pad = np.zeros(self._half, dtype=np.float64)
        return self.process(pad)


def refine_falling_edge(freq: np.ndarray, run_start: int, fs: float) -> int:
    """Refine the start of a detected low run to the true transition point.

    The FIR Hilbert transformer smears a frequency step into a ramp of
    roughly +-half the filter length, so the band-crossing point where a
    run is first detected sits *after* the true edge.  The ramp's midpoint
    crossing, however, lands on the true edge, so we estimate the tone
    levels on either side and walk back to their midpoint crossing.
    All indices are local to ``freq``.
    """
    n = len(freq)
    ms = lambda t: int(round(t * fs / 1000.0))  # noqa: E731
    pre_lo = max(0, run_start - ms(16))
    pre_hi = max(pre_lo, run_start - ms(8))
    post_lo = min(n, run_start + ms(6))
    post_hi = min(n, run_start + ms(9))
    if pre_hi <= pre_lo or post_hi <= post_lo:
        return run_start
    pre = float(np.percentile(freq[pre_lo:pre_hi], 75))
    post = float(np.median(freq[post_lo:post_hi]))
    mid = 0.5 * (pre + post)
    i = run_start - 1
    last = run_start
    limit = max(0, run_start - ms(25))
    while i >= limit:
        if freq[i] <= mid:
            last = i
            i -= 1
        else:
            break
    return last


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average, same length as the input.

    Near the edges the window shrinks (mean of whatever is available),
    which avoids the dips that zero padding would introduce.
    """
    x = np.asarray(x, dtype=np.float64)
    if k <= 1 or x.size == 0:
        return x.copy()
    n = x.size
    half = k // 2
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)

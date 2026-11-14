"""IQ receive chain: doppler-corrected FM demodulation to baseband audio.

The SDR is parked at a *fixed* center frequency (nominal - tuning offset).
A per-sample complex NCO removes the tuning offset, the predicted doppler
shift and any residual frequency error, dropping the satellite signal to
DC.  A channel low-pass then selects the FM signal, the complex signal is
decimated to the audio rate, and a conjugate-product discriminator recovers
the FM audio.  Software doppler correction is sample-accurate and glitch-
free, unlike periodic hardware retuning.

Signal flow (fs_iq = 240 kHz, fs_audio = 48 kHz):

    IQ --NCO(offset+doppler)--> complex LPF 15 kHz --> /5 --> FM disc -->
    DC block --> de-emphasis (optional) --> normalized float audio
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from scipy.signal import firwin, lfilter

log = logging.getLogger(__name__)

FS_IQ = 240_000        # RTL-SDR sample rate (240 kHz keeps a 5:1 ratio to audio)
FS_AUDIO = 48_000
CHANNEL_BW_HZ = 15_000  # complex low-pass cutoff around DC
CHANNEL_TAPS = 127
DISC_SCALE_HZ = 4_000.0  # discriminator Hz that maps to full-scale audio
DC_BLOCK_HZ = 250.0
DEEMPH_TAU_S = 750e-6    # standard narrow-FM de-emphasis time constant


class OnePole:
    """Streaming one-pole low-pass (used for DC block and de-emphasis)."""

    def __init__(self, fs: float, fc: float):
        self._a = float(np.exp(-2.0 * np.pi * fc / fs))
        self._b = 1.0 - self._a
        self._zi = np.zeros(1)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return x
        y, self._zi = lfilter([self._b], [1.0, -self._a], x, zi=self._zi)
        return y


class Preemphasis:
    """Streaming +6 dB/octave pre-emphasis (transmit side, for the simulator)."""

    def __init__(self, fs: float, tau_s: float = DEEMPH_TAU_S):
        self.lp = OnePole(fs, 1.0 / (2.0 * np.pi * tau_s))

    def process(self, x: np.ndarray) -> np.ndarray:
        return x - self.lp.process(x)


class FMModulator:
    """Streaming FM modulator (transmit side, for the simulator and tests).

    Consumes baseband audio and a per-sample carrier-offset array (tuning
    offset + doppler), both at ``fs_iq``; produces complex IQ with
    continuous phase.
    """

    def __init__(self, fs_iq: float = FS_IQ, deviation_hz: float = 3500.0):
        self.fs = float(fs_iq)
        self.deviation_hz = float(deviation_hz)
        self._phase = 0.0

    def process(self, audio: np.ndarray, offset_hz: np.ndarray) -> np.ndarray:
        inst = self.deviation_hz * np.asarray(audio, dtype=np.float64) \
            + np.asarray(offset_hz, dtype=np.float64)
        dphi = 2.0 * np.pi * inst / self.fs
        phase = self._phase + np.concatenate(
            [[0.0], np.cumsum(0.5 * (dphi[:-1] + dphi[1:]))])
        self._phase = phase[-1] + dphi[-1]
        return np.exp(1j * phase).astype(np.complex64)


class ReceiveChain:
    """Streaming IQ -> audio converter with software doppler correction."""

    def __init__(self,
                 fs_iq: float = FS_IQ,
                 fs_audio: float = FS_AUDIO,
                 tuning_offset_hz: float = 20_000.0,
                 fine_offset_hz: float = 0.0,
                 deemphasis: bool = True,
                 channel_bw_hz: float = CHANNEL_BW_HZ,
                 doppler_segment: Optional[Callable[[float, float, int], np.ndarray]] = None):
        self.fs_iq = float(fs_iq)
        self.fs_audio = float(fs_audio)
        self.ratio = int(round(self.fs_iq / self.fs_audio))
        self.tuning_offset_hz = float(tuning_offset_hz)
        self.fine_offset_hz = float(fine_offset_hz)
        self.deemphasis = deemphasis
        self.doppler_segment = doppler_segment

        self._taps = firwin(CHANNEL_TAPS, channel_bw_hz, fs=self.fs_iq)
        self._zi = np.zeros(CHANNEL_TAPS - 1, dtype=np.complex128)
        self._nco_phase = 0.0
        self._decim_phase = 0
        self._rem = np.zeros(0, dtype=np.complex64)
        self._rem_t = 0.0
        self._dc_lp = OnePole(self.fs_audio, DC_BLOCK_HZ)
        if deemphasis:
            self._deemph_lp = OnePole(self.fs_audio, 1.0 / (2.0 * np.pi * DEEMPH_TAU_S))

    # ------------------------------------------------------------------
    def _doppler(self, t0: float, t1: float, n: int) -> np.ndarray:
        if self.doppler_segment is None:
            return np.zeros(n)
        return np.asarray(self.doppler_segment(t0, t1, n), dtype=np.float64)

    def process(self, iq: np.ndarray, t_start: float) -> np.ndarray:
        """Process one IQ chunk; ``t_start`` is the unix time of sample 0."""
        iq = np.asarray(iq, dtype=np.complex64).ravel()
        if iq.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._rem.size:
            work = np.concatenate([self._rem, iq])
            t_work = self._rem_t
        else:
            work = iq
            t_work = t_start
        # split off a whole number of decimation blocks
        usable = (work.size // self.ratio) * self.ratio
        self._rem = work[usable:].copy()
        self._rem_t = t_work + usable / self.fs_iq
        work = work[:usable]
        if work.size == 0:
            return np.zeros(0, dtype=np.float32)

        n = work.size
        t_end = t_work + (n - 1) / self.fs_iq
        dop = self._doppler(t_work, t_end, n)
        f_nco = self.tuning_offset_hz + self.fine_offset_hz + dop

        # continuous-phase NCO (trapezoidal phase integral)
        dphi = 2.0 * np.pi * f_nco / self.fs_iq
        phase = self._nco_phase + np.concatenate(
            [[0.0], np.cumsum(0.5 * (dphi[:-1] + dphi[1:]))])
        self._nco_phase = phase[-1] + dphi[-1]
        corrected = work * np.exp(-1j * phase)

        # channel low-pass (complex signal, real symmetric taps)
        filtered, self._zi = lfilter(self._taps, 1.0, corrected, zi=self._zi)

        # decimate to the audio rate
        dec = filtered[self._decim_phase::self.ratio]
        self._decim_phase = (self._decim_phase + n) % self.ratio

        # FM discriminator (conjugate product keeps phase increments small)
        if dec.size < 2:
            return np.zeros(0, dtype=np.float32)
        dz = dec[1:] * np.conj(dec[:-1])
        audio_hz = np.angle(dz) * (self.fs_audio / (2.0 * np.pi))

        # DC block, de-emphasis, normalization
        audio = audio_hz - self._dc_lp.process(audio_hz)
        if self.deemphasis:
            audio = self._deemph_lp.process(audio)
        audio = np.clip(audio / DISC_SCALE_HZ, -1.0, 1.0)
        return audio.astype(np.float32)

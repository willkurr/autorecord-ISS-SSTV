"""Recording pipeline: wires a source through the DSP chain to consumers.

Everything heavy runs on the source thread: the sink callback performs
doppler-corrected demodulation, feeds the SSTV engine, writes the
recorder, and pushes spectrum/level updates.  The GUI receives updates
via Qt signals (queued connections make cross-thread emission safe).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from core.doppler import DopplerModel
from core.dsp import ReceiveChain
from core.recorder import PassRecorder
from core.settings import Settings
from core.sources.base import Source
from core.sources.sim import SimulatorSource
from core.sources.soundcard import SoundcardSource
from core.tracker import PassInfo, Tracker
from sstv.engine import SSTVEngine
from sstv.events import EventType, SSTVEvent
from util import RECORDINGS_DIR

log = logging.getLogger(__name__)

SPECTRUM_BINS = 512


class AudioReservoir:
    """Jitter-absorbing sample buffer between the DSP thread and the
    sound-card callback.

    The DSP produces audio in bursts (one ~100 ms chunk at a time) while
    the callback consumes a fixed small block at exact intervals.  Feeding
    the callback straight from a chunk queue causes stutter on every
    scheduling hiccup (and, worse, can replay chunk remainders out of
    order).  The reservoir decouples the two: writers append; the callback
    pulls exactly what it needs.  Playback only starts once the reservoir
    holds a small prebuffer, and an underrun re-primes it so a hiccup
    costs one dropout instead of a sustained choppy cycle.
    """

    PREBUFFER_S = 0.30   # playback starts at this fill level; also the
                         # jitter tolerance for bursty chunk production
    MAX_BUFFER_S = 1.0   # bound latency; older audio is dropped beyond this

    def __init__(self, fs: float,
                 prebuffer_s: float = PREBUFFER_S,
                 max_buffer_s: float = MAX_BUFFER_S):
        self.fs = float(fs)
        self._prebuffer = max(1, int(prebuffer_s * fs))
        self._maxlen = max(self._prebuffer, int(max_buffer_s * fs))
        self._buf = np.zeros(0, dtype=np.float32)
        self._primed = False
        self._lock = threading.Lock()
        self.underruns = 0

    def write(self, audio: np.ndarray) -> None:
        """Append samples (called from the DSP thread)."""
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if audio.size == 0:
            return
        with self._lock:
            self._buf = np.concatenate([self._buf, audio])
            if len(self._buf) > self._maxlen:
                self._buf = self._buf[-self._maxlen:]

    def pull(self, frames: int) -> np.ndarray:
        """Take exactly ``frames`` samples (called from the audio thread).

        Returns zeros while the prebuffer is filling and for whatever the
        reservoir cannot supply during an underrun.
        """
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            if not self._primed:
                if len(self._buf) < self._prebuffer:
                    return out
                self._primed = True
            n = min(frames, len(self._buf))
            if n == 0:
                # fully drained: wait for a fresh prebuffer before resuming
                self._primed = False
                self.underruns += 1
                return out
            out[:n] = self._buf[:n]
            self._buf = self._buf[n:]
            if n < frames:
                # partial frame: play what is there; the next burst is at
                # most one chunk away, so keep playing (no re-prime) -
                # re-priming would turn every hiccup into a long gap
                self.underruns += 1
        return out

    def reset(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            self._primed = False


class AudioMonitor:
    """Plays the demodulated audio to the default output (best effort)."""

    def __init__(self, fs: float = 48_000.0):
        self.fs = float(fs)
        self._stream = None
        self.reservoir = AudioReservoir(fs)
        try:
            import sounddevice

            def callback(outdata, frames, time_info, status):
                outdata[:, 0] = self.reservoir.pull(frames)

            try:
                # 'high' gives the device buffer room to ride out GIL
                # contention with the GUI (callback delays -> pops otherwise)
                self._stream = sounddevice.OutputStream(
                    samplerate=int(self.fs), channels=1, dtype="float32",
                    blocksize=0, latency="high", callback=callback)
            except Exception:  # noqa: BLE001 - some devices dislike 'high'
                self._stream = sounddevice.OutputStream(
                    samplerate=int(self.fs), channels=1, dtype="float32",
                    blocksize=0, callback=callback)
            self._stream.start()
            log.info("Audio monitoring enabled")
        except Exception as exc:  # noqa: BLE001 - monitoring is optional
            log.info("Audio monitoring unavailable: %s", exc)
            self._stream = None

    def write(self, audio: np.ndarray) -> None:
        # buffer even without a stream: if the device appears later the
        # audio continues seamlessly; the reservoir bounds its own latency
        self.reservoir.write(audio)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None


class PassPipeline(QObject):
    """One recording session from start() to stop()."""

    sstvEvent = Signal(object)          # SSTVEvent
    spectrumRow = Signal(object)        # np.ndarray (uint8 RGB row) or None
    levelUpdate = Signal(float)         # rms level 0..1
    dopplerUpdate = Signal(float)       # current doppler, Hz
    sourceDone = Signal()               # the source stream ended
    error = Signal(str)

    def __init__(self, settings: Settings, tracker: Tracker,
                 pass_info: Optional[PassInfo], image_store,
                 satellite_name: str = "ISS"):
        super().__init__()
        self.settings = settings
        self.tracker = tracker
        self.pass_info = pass_info
        self.image_store = image_store
        self.satellite_name = satellite_name
        self._source: Optional[Source] = None
        self._chain: Optional[ReceiveChain] = None
        self._engine: Optional[SSTVEngine] = None
        self._recorder: Optional[PassRecorder] = None
        self._monitor: Optional[AudioMonitor] = None
        self._spectrum_state = 0.0
        self._doppler_model = None
        self.running = False
        self.result: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _build_source(self) -> Source:
        s = self.settings
        if s.source_type == "sim":
            if self.pass_info is None:
                raise RuntimeError("no pass selected for simulation")
            return SimulatorSource(
                self.tracker, self.pass_info,
                freq_hz=s.nominal_freq_hz,
                tuning_offset_hz=s.tuning_offset_hz,
                snr_db=s.sim_snr_db,
                speed=s.sim_speed,
                preroll_s=0.0, postroll_s=0.0,
                fs_iq=240_000.0, chunk_samples=12_000)
        if s.source_type == "soundcard":
            return SoundcardSource(fs=48_000.0, chunk_samples=4_800)
        from core.sources.sdr import RtlSdrSource
        center = s.nominal_freq_hz - s.tuning_offset_hz
        gain = "auto" if s.gain_auto else s.manual_gain_db
        return RtlSdrSource(center_freq_hz=center, fs=240_000.0,
                            chunk_samples=12_000, ppm=s.ppm, gain=gain)

    def _build_chain(self, source: Source) -> Optional[ReceiveChain]:
        """Receive chain for IQ sources; None for already-audio sources."""
        s = self.settings
        if not source.is_iq:
            return None
        doppler = DopplerModel(self.tracker, s.nominal_freq_hz,
                               enabled=s.doppler_correction)
        if isinstance(source, SimulatorSource):
            # simulated pass time can differ from wall time (speed != 1 or
            # a mid-pass start): map the chain's doppler through it
            def segment(t0, t1, n):
                return doppler.segment(source.sim_time(t0), source.sim_time(t1), n)
        else:
            segment = doppler.segment
        self._doppler_model = doppler
        return ReceiveChain(
            tuning_offset_hz=s.tuning_offset_hz,
            fine_offset_hz=s.fine_offset_hz,
            deemphasis=s.deemphasis,
            channel_bw_hz=s.channel_bw_hz,
            doppler_segment=segment)

    # ------------------------------------------------------------------
    def start(self) -> None:
        s = self.settings
        self._source = self._build_source()

        tle = self.tracker.tle
        meta = {
            "source": s.source_type,
            "tle_name": tle.name,
            "tle_line1": tle.line1,
            "tle_line2": tle.line2,
            "station": {"lat": s.lat_deg, "lon": s.lon_deg, "alt_m": s.alt_m},
            "ppm": s.ppm, "gain": "auto" if s.gain_auto else s.manual_gain_db,
            "doppler_correction": s.doppler_correction,
            "deemphasis": s.deemphasis,
            "pass": None,
        }
        if self.pass_info is not None:
            meta["pass"] = {
                "aos": self.pass_info.aos.isoformat(),
                "los": self.pass_info.los.isoformat(),
                "max_elevation_deg": self.pass_info.max_el_deg,
            }
        self._recorder = PassRecorder(
            directory=RECORDINGS_DIR,
            satellite=self.satellite_name,
            freq_hz=s.nominal_freq_hz, fs=48_000.0, meta=meta)

        self._chain = self._build_chain(self._source)
        self._engine = SSTVEngine(48_000.0, on_event=self._on_sstv_event)
        if s.monitor_audio:
            self._monitor = AudioMonitor(48_000.0)

        self._recorder.start()
        self.running = True
        self._source.on_end = lambda: self.sourceDone.emit()
        self._source.start(self._sink)
        log.info("Pipeline started (source=%s)", s.source_type)

    def stop(self) -> Dict[str, Any]:
        if not self.running:
            return self.result
        self.running = False
        source_error = None
        try:
            if self._source is not None:
                self._source.stop()
        except Exception as exc:  # noqa: BLE001
            source_error = str(exc)
            log.warning("Source stop failed: %s", exc)
        if self._engine is not None:
            try:
                self._engine.flush()
            except Exception:  # noqa: BLE001
                log.exception("engine flush failed")
        wav_path = None
        if self._recorder is not None:
            try:
                wav_path = self._recorder.close()
            except Exception:  # noqa: BLE001
                log.exception("recorder close failed")
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None
        self.result = {
            "wav_path": wav_path,
            "images": len(self._recorder._images) if self._recorder else 0,
            "pass": self.pass_info,
            "error": source_error,
        }
        self._source = None
        self._chain = None
        self._engine = None
        self._recorder = None
        log.info("Pipeline stopped: %s", self.result["wav_path"])
        return self.result

    # ------------------------------------------------------------------
    def _on_sstv_event(self, ev: SSTVEvent) -> None:
        if ev.type == EventType.IMAGE_COMPLETE and ev.image is not None:
            try:
                rec_name = None
                if self._recorder is not None:
                    rec_name = self._recorder.wav_path.name
                path = self.image_store.save(
                    ev.image,
                    meta={"mode": ev.mode.name if ev.mode else None,
                          "truncated": ev.truncated,
                          "sync_quality": ev.sync_quality,
                          "recording": rec_name,
                          "satellite": self.satellite_name,
                          "frequency_hz": self.settings.nominal_freq_hz,
                          "source": self.settings.source_type})
                if self._recorder is not None:
                    self._recorder.note_image(path.name, {
                        "mode": ev.mode.name if ev.mode else None,
                        "truncated": ev.truncated,
                        "sync_quality": ev.sync_quality})
            except Exception:  # noqa: BLE001
                log.exception("saving decoded image failed")
        try:
            self.sstvEvent.emit(ev)
        except Exception:  # noqa: BLE001
            log.exception("sstv event emit failed")

    # ------------------------------------------------------------------
    def _sink(self, chunk) -> None:
        if not self.running:
            return
        try:
            if chunk.is_iq and self._chain is not None:
                audio = self._chain.process(chunk.data, chunk.t)
            else:
                audio = np.asarray(chunk.data, dtype=np.float32)
            if audio.size == 0:
                return
            if self._recorder is not None:
                self._recorder.write(audio)
            if self._engine is not None:
                self._engine.process(audio)
            if self._monitor is not None:
                self._monitor.write(audio)
            self._emit_spectrum(audio)
        except Exception:  # noqa: BLE001 - never kill the source thread
            log.exception("pipeline sink error")

    # ------------------------------------------------------------------
    def _emit_spectrum(self, audio: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        try:
            self.levelUpdate.emit(rms)
        except Exception:  # noqa: BLE001
            pass
        now = time.time()
        if now - self._spectrum_state < 0.05:  # ~20 rows/s
            return
        self._spectrum_state = now
        try:
            spec = np.abs(np.fft.rfft(audio[:4096] * np.hanning(min(len(audio), 4096))))
            row = self._spec_to_row(spec)
            self.spectrumRow.emit(row)
        except Exception:  # noqa: BLE001
            log.exception("spectrum update failed")
        if self._doppler_model is not None:
            try:
                self.dopplerUpdate.emit(self._doppler_model.doppler_at(now))
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _spec_to_row(spec: np.ndarray) -> np.ndarray:
        # map 0..4 kHz of the 48 kHz spectrum onto SPECTRUM_BINS columns
        freqs = np.linspace(0, 4000, SPECTRUM_BINS)
        bin_hz = 48000.0 / 2.0 / (len(spec) - 1)
        idx = np.clip(np.round(freqs / bin_hz).astype(int), 0, len(spec) - 1)
        mag = 20.0 * np.log10(np.maximum(spec[idx], 1e-9))
        v = np.clip((mag + 60.0) / 60.0, 0.0, 1.0)
        # simple blue->green->yellow->red colormap
        r = np.clip(1.6 * v - 0.6, 0, 1)
        g = np.clip(np.where(v < 0.5, 2.0 * v, 2.0 - 2.0 * v), 0, 1)
        b = np.clip(1.0 - 2.0 * v, 0, 1)
        row = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)
        return row

"""SSTV decoding engine: glues tone analysis, VIS detection and mode decoding.

Feed it audio chunks (any size, any time); it emits events:
IMAGE_START -> LINE* -> IMAGE_COMPLETE.  Works the same for live streams
and offline files.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

from sstv.detector import VISDetector
from sstv.events import EventType, SSTVEvent
from sstv.freq import StreamingAnalytic, StreamingBandpass, ToneFrame

log = logging.getLogger(__name__)

# importing robot36 registers the mode in the registry
import sstv.robot36  # noqa: F401  (pylint: disable=unused-import)


class SSTVEngine:
    """Streaming SSTV detector/decoder for one audio stream."""

    def __init__(self, fs: float = 48000.0,
                 on_event: Optional[Callable[[SSTVEvent], None]] = None):
        self.fs = float(fs)
        self.on_event = on_event
        self.bandpass = StreamingBandpass(fs)
        self.analytic = StreamingAnalytic(fs)
        self.detector = VISDetector(fs)
        self.active = None  # current line decoder

    def process(self, audio: np.ndarray) -> List[SSTVEvent]:
        audio = np.asarray(audio, dtype=np.float64).ravel()
        if audio.size == 0:
            return []
        audio = self.bandpass.process(audio)
        frame: ToneFrame = self.analytic.process(audio)
        events: List[SSTVEvent] = []

        if self.active is not None and not self.active.done:
            events += self.active.process(frame)
            if self.active.done:
                self.active = None
                # the remainder of this frame may already hold the next
                # header; let the detector look at it
                det = self.detector.process(frame)
                if det is not None:
                    events += self._start_image(det, frame)
        elif self.active is None:
            det = self.detector.process(frame)
            if det is not None:
                events += self._start_image(det, frame)

        if self.on_event is not None:
            for ev in events:
                try:
                    self.on_event(ev)
                except Exception:  # noqa: BLE001 - never kill the stream
                    log.exception("SSTV event handler failed")
        return events

    def flush(self) -> List[SSTVEvent]:
        """End of stream: finalize any in-progress image."""
        events: List[SSTVEvent] = []
        # Drain the band-pass and analytic delay lines (their group delays
        # hold the tail of the real signal) so the last line can decode.
        pad = np.zeros(1024, dtype=np.float64)
        tail = self.analytic.process(self.bandpass.process(pad))
        tail2 = self.analytic.flush()
        for frame in (tail, tail2):
            if frame.freq.size == 0:
                continue
            if self.active is not None and not self.active.done:
                events += self.active.process(frame)
                if self.active.done:
                    self.active = None
            else:
                det = self.detector.process(frame)
                if det is not None:
                    events += self._start_image(det, frame)
        if self.active is not None and not self.active.done:
            events += self.active.flush()
            self.active = None
        if self.on_event is not None:
            for ev in events:
                try:
                    self.on_event(ev)
                except Exception:  # noqa: BLE001
                    log.exception("SSTV event handler failed")
        return events

    def _start_image(self, det, frame: ToneFrame) -> List[SSTVEvent]:
        events = [SSTVEvent(EventType.IMAGE_START, mode=det.mode,
                            start_index=det.data_start)]
        self.active = det.mode.decoder_factory(self.fs, det.data_start)
        events += self.active.process(frame)
        if self.active.done:
            self.active = None
        return events


def decode_file(path, chunk_samples: int = 48000,
                on_event: Optional[Callable[[SSTVEvent], None]] = None
                ) -> List[SSTVEvent]:
    """Convenience: decode all images in a WAV/PCM file; returns COMPLETE events."""
    import wave

    completed: List[SSTVEvent] = []

    def cb(ev: SSTVEvent) -> None:
        if on_event is not None:
            on_event(ev)
        if ev.type == EventType.IMAGE_COMPLETE:
            completed.append(ev)

    with wave.open(str(path), "rb") as wav:
        fs = wav.getframerate()
        n_ch = wav.getnchannels()
        width = wav.getsampwidth()
        if width != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        engine = SSTVEngine(fs=fs, on_event=cb)
        while True:
            raw = wav.readframes(chunk_samples)
            if not raw:
                break
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if n_ch > 1:
                data = data.reshape(-1, n_ch).mean(axis=1)
            engine.process(data)
    engine.flush()
    return completed

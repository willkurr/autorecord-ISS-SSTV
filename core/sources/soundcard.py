"""Sound-card (line-in / microphone) source.

For testing the decoder with any receiver, or replaying SSTV audio into
the application without an SDR.  sounddevice/PortAudio is imported
lazily because it needs a native library that may be absent.
"""
from __future__ import annotations

import logging
import queue
import time
from typing import Callable, Optional

import numpy as np

from core.sources.base import Chunk, SourceError, TimedSource

log = logging.getLogger(__name__)


class SoundcardSource(TimedSource):
    """Streams mono float audio from a sound-card input."""

    def __init__(self,
                 fs: float = 48_000.0,
                 chunk_samples: int = 4_800,
                 device: Optional[int] = None,
                 label: str = "soundcard"):
        super().__init__(fs, chunk_samples, is_iq=False, label=label)
        self.device = device
        self._stream = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

    # ------------------------------------------------------------------
    def _loop(self, sink: Callable[[Chunk], None]) -> None:
        try:
            import sounddevice  # lazy: PortAudio may be absent
        except Exception as exc:  # noqa: BLE001
            raise SourceError(
                "sounddevice/PortAudio is not available. "
                "Install PortAudio (Linux: sudo apt install libportaudio2; "
                "macOS: brew install portaudio; Windows: bundled).\n"
                f"({exc})"
            ) from exc

        def callback(indata, frames, time_info, status):
            if status:
                log.debug("soundcard status: %s", status)
            try:
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # drop rather than block the audio thread

        try:
            self._stream = sounddevice.InputStream(
                samplerate=int(self.fs), channels=1, dtype="float32",
                blocksize=self.chunk_samples, callback=callback)
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"Could not open sound input: {exc}") from exc

        log.info("Sound input open at %.0f Hz", self.fs)
        try:
            while not self._stop_evt.is_set():
                try:
                    block = self._queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                t = self._stamp_n(len(block))
                sink(Chunk(data=block.astype(np.float32), t=t,
                           is_iq=False, fs=self.fs))
        finally:
            self._cleanup()

    def _stamp_n(self, n: int) -> float:
        if self._t0 is None:
            self._t0 = time.time()
        t = self._t0 + self._samples / self.fs
        self._samples += n
        return t

    def _cleanup(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

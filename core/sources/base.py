"""Source abstraction: streaming chunks of IQ or audio into a sink."""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """Raised when a source cannot be opened or configured."""


@dataclass
class Chunk:
    """A block of samples from a source.

    ``t`` is the unix timestamp of the first sample; ``is_iq`` selects
    complex IQ (needs the receive chain) vs. real audio (already
    demodulated by an external receiver).
    """
    data: np.ndarray
    t: float
    is_iq: bool
    fs: float


class Source(ABC):
    """A streaming signal source that pushes chunks to a sink callback."""

    def __init__(self, fs: float, chunk_samples: int, is_iq: bool,
                 label: str = "source"):
        self.fs = float(fs)
        self.chunk_samples = int(chunk_samples)
        self.is_iq = is_iq
        self.label = label
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._error: Optional[Exception] = None
        self.on_end: Optional[Callable[[], None]] = None  # called when the stream ends

    # ------------------------------------------------------------------
    def start(self, sink: Callable[[Chunk], None]) -> None:
        if self._thread is not None:
            raise SourceError("source already running")
        self._stop_evt.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, args=(sink,),
                                        name=f"src-{self.label}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._error is not None:
            raise self._error

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _run(self, sink: Callable[[Chunk], None]) -> None:
        try:
            self._loop(sink)
        except Exception as exc:  # noqa: BLE001 - surface to stop()
            self._error = exc
            log.exception("Source %s failed", self.label)
        finally:
            try:
                self._cleanup()
            finally:
                if self.on_end is not None:
                    try:
                        self.on_end()
                    except Exception:  # noqa: BLE001
                        log.exception("source end callback failed")

    @abstractmethod
    def _loop(self, sink: Callable[[Chunk], None]) -> None:
        ...

    def _cleanup(self) -> None:
        pass


class TimedSource(Source):
    """Base for sources that timestamp chunks from a sample counter.

    The first sample of the stream is stamped with the wall clock when the
    loop starts; later chunks derive from the running sample count, which
    is immune to scheduling jitter.
    """

    def __init__(self, fs: float, chunk_samples: int, is_iq: bool, label: str):
        super().__init__(fs, chunk_samples, is_iq, label)
        self._t0: Optional[float] = None
        self._samples = 0

    def _stamp(self) -> float:
        if self._t0 is None:
            self._t0 = time.time()
        t = self._t0 + self._samples / self.fs
        self._samples += self.chunk_samples
        return t

    def _pace(self, wall_target: float, speed: float = 1.0) -> None:
        """Sleep so the source produces samples no faster than real time.

        ``wall_target`` is the wall-clock time this chunk should be
        produced at; ``speed`` divides the wait for accelerated sources.
        """
        if speed <= 0:
            return
        while not self._stop_evt.is_set():
            delay = (wall_target - time.time()) / speed
            if delay <= 0:
                return
            time.sleep(min(delay, 0.25))

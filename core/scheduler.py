"""Pass scheduler: arms a pass and starts/stops recording automatically."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from core.pipeline import PassPipeline
from core.settings import Settings
from core.tracker import PassInfo

log = logging.getLogger(__name__)


class PassScheduler(QObject):
    """Watches the clock and drives one PassPipeline per recording."""

    recordingStarted = Signal(object)   # PassInfo or None
    recordingStopped = Signal(object)   # pipeline result dict
    statusChanged = Signal(str)

    def __init__(self, settings: Settings,
                 pipeline_factory: Callable[[Optional[PassInfo]], PassPipeline],
                 parent=None):
        super().__init__(parent)
        self.settings = settings
        self.pipeline_factory = pipeline_factory
        self.armed_pass: Optional[PassInfo] = None
        self.pipeline: Optional[PassPipeline] = None
        self.auto_record = settings.auto_record
        self._source_done = False
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.tick)
        self._timer.start()

    def _on_source_done(self) -> None:
        self._source_done = True

    # ------------------------------------------------------------------
    def arm_pass(self, pass_info: Optional[PassInfo]) -> None:
        self.armed_pass = pass_info
        if pass_info is None:
            self._status("No pass armed")
        elif self.pipeline is not None:
            self._status(f"Recording (armed next: {self._fmt(pass_info)})")
        else:
            self._status(f"Armed {self._fmt(pass_info)}")

    def set_auto_record(self, enabled: bool) -> None:
        self.auto_record = enabled
        if enabled and self.armed_pass is not None:
            self.tick()  # possibly already inside the window

    # ------------------------------------------------------------------
    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _fmt(p: PassInfo) -> str:
        return p.aos.strftime("%Y-%m-%d %H:%M:%SZ")

    def _status(self, msg: str) -> None:
        self.statusChanged.emit(msg)

    # ------------------------------------------------------------------
    def tick(self) -> None:
        now = self._now()
        if self.pipeline is not None:
            if self._source_done:
                # the source stream ended on its own (e.g. a simulated pass
                # finished); wrap up without waiting for the clock
                self.stop_recording()
                return
            p = self.pipeline.pass_info
            if p is not None:
                stop_at = p.los.timestamp() + self.settings.postroll_s
                if now.timestamp() >= stop_at:
                    self.stop_recording()
            return
        if not self.auto_record or self.armed_pass is None:
            return
        p = self.armed_pass
        start_at = p.aos.timestamp() - self.settings.preroll_s
        if now.timestamp() >= start_at:
            self.start_recording(p)

    # ------------------------------------------------------------------
    def start_recording(self, pass_info: Optional[PassInfo]) -> None:
        if self.pipeline is not None:
            self._status("Already recording")
            return
        try:
            pipeline = self.pipeline_factory(pass_info)
            pipeline.sourceDone.connect(self._on_source_done)
            pipeline.start()
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to start recording")
            self._status(f"Record start failed: {exc}")
            return
        self._source_done = False
        self.pipeline = pipeline
        self.recordingStarted.emit(pass_info)
        if pass_info is not None:
            self._status(f"Recording pass {self._fmt(pass_info)}")
        else:
            self._status("Recording (manual)")

    def stop_recording(self) -> None:
        if self.pipeline is None:
            return
        pipeline = self.pipeline
        self.pipeline = None
        try:
            result = pipeline.stop()
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to stop recording")
            result = {"wav_path": None, "images": 0, "pass": pipeline.pass_info,
                      "error": str(exc)}
        self.recordingStopped.emit(result)
        self._status("Recording stopped")

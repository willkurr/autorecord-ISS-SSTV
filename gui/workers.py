"""Small helpers for running background jobs from the GUI."""
from __future__ import annotations

import logging
import threading
from typing import Callable

from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class JobRunner(QObject):
    """Runs a callable in a Python thread and reports back via Qt signals."""

    succeeded = Signal(object)
    failed = Signal(str)

    def run(self, fn: Callable[[], object]) -> None:
        def wrapper():
            try:
                self.succeeded.emit(fn())
            except Exception as exc:  # noqa: BLE001
                log.exception("background job failed")
                self.failed.emit(str(exc))

        threading.Thread(target=wrapper, daemon=True).start()

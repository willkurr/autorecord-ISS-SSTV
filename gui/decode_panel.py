"""Live SSTV decode panel: one card per detected image, updated per line."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QVBoxLayout, QWidget)

from sstv.encoder import HEIGHT, WIDTH
from sstv.events import EventType, SSTVEvent


class DecodeCard(QFrame):
    """A single live-decoding image."""

    def __init__(self, mode_name: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self._mode = mode_name
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.image_label = QLabel()
        self.image_label.setMinimumSize(WIDTH, HEIGHT)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: black;")
        layout.addWidget(self.image_label)
        self.status_label = QLabel(f"{mode_name} - waiting for sync")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        self._buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self._qimg = QImage(self._buffer.data, WIDTH, HEIGHT, 3 * WIDTH,
                            QImage.Format_RGB888)
        self._lines_done = 0
        self._done = False

    # ------------------------------------------------------------------
    def set_line(self, line: int, row: np.ndarray) -> None:
        if 0 <= line < HEIGHT and len(row) == WIDTH:
            self._buffer[line] = row
        self._lines_done = max(self._lines_done, line + 1)
        self._refresh()
        self.status_label.setText(f"{self._mode} - line {self._lines_done}/{HEIGHT}")

    def set_complete(self, image: np.ndarray, truncated: bool,
                     sync_quality: float) -> None:
        if image is not None and image.shape == self._buffer.shape:
            self._buffer[:] = image
        self._done = True
        self._refresh()
        pct = int(round(sync_quality * 100))
        state = "partial" if truncated else "complete"
        self.status_label.setText(
            f"{self._mode} - {state} ({self._lines_done}/{HEIGHT} lines, sync {pct}%)")
        self.setStyleSheet("QFrame { border: 1px solid #3a7d44; }")

    def _refresh(self) -> None:
        self._qimg = QImage(self._buffer.data, WIDTH, HEIGHT, 3 * WIDTH,
                            QImage.Format_RGB888)
        self.image_label.setPixmap(
            QPixmap.fromImage(self._qimg).scaled(
                self.image_label.size(), Qt.KeepAspectRatio,
                Qt.FastTransformation))


class DecodePanel(QWidget):
    """Horizontal row of live decode cards."""

    imageSaved = Signal(object)  # not used directly; pipeline saves images

    MAX_CARDS = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards_layout = QHBoxLayout()
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.addStretch(1)
        inner = QWidget()
        inner.setLayout(self._cards_layout)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(HEIGHT + 60)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        title = QLabel("Live SSTV decode")
        title.setStyleSheet("font-weight: bold;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)
        layout.addWidget(scroll)
        self._cards = {}
        self._counter = 0

    # ------------------------------------------------------------------
    def handle_event(self, ev: SSTVEvent) -> None:
        if ev.type == EventType.IMAGE_START:
            self._counter += 1
            card = DecodeCard(ev.mode.name if ev.mode else "SSTV")
            self._cards[self._counter] = card
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
            self._prune()
        elif ev.type == EventType.LINE and ev.row is not None:
            card = self._cards.get(self._counter)
            if card is not None:
                card.set_line(ev.line, ev.row)
        elif ev.type == EventType.IMAGE_COMPLETE:
            card = self._cards.get(self._counter)
            if card is not None:
                card.set_complete(ev.image, ev.truncated, ev.sync_quality)

    def _prune(self) -> None:
        # drop the oldest cards beyond MAX_CARDS (keep the newest)
        keys = sorted(self._cards)
        while len(keys) > self.MAX_CARDS:
            old = self._cards.pop(keys[0])
            old.setParent(None)
            old.deleteLater()
            keys = sorted(self._cards)

    def clear_finished_pass(self) -> None:
        for k in list(self._cards):
            card = self._cards.pop(k)
            card.setParent(None)
            card.deleteLater()
        self._counter = 0

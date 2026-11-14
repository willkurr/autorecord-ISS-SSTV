"""Scrolling audio spectrum waterfall, 0-4 kHz."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class Waterfall(QWidget):
    """Displays spectrum rows pushed in from the pipeline thread."""

    ROWS = 180
    COLS = 512

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._data = np.zeros((self.ROWS, self.COLS, 3), dtype=np.uint8)
        self._image = QImage(self._data.data, self.COLS, self.ROWS,
                             3 * self.COLS, QImage.Format_RGB888)

    # ------------------------------------------------------------------
    def push_row(self, row: np.ndarray) -> None:
        """row: (COLS, 3) uint8 RGB. Called from any thread."""
        if row is None or len(row) != self.COLS:
            return
        # scroll down by one
        self._data[:-1] = self._data[1:]
        self._data[-1] = row
        self._image = QImage(self._data.data, self.COLS, self.ROWS,
                             3 * self.COLS, QImage.Format_RGB888)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(8, 10, 16))
        target = QRectF(self.rect()).adjusted(0, 14, 0, -2)
        p.drawImage(target, self._image)
        # frequency scale
        font = QFont(p.font())
        font.setPointSize(8)
        p.setFont(font)
        p.setPen(QColor(150, 165, 190))
        w = self.width()
        for khz in range(0, 5):
            x = int(khz / 4.0 * w)
            p.drawLine(x, 12, x, 16)
            p.drawText(QRectF(x - 20, 0, 40, 12), Qt.AlignCenter, f"{khz}k")
        p.setPen(QPen(QColor(60, 75, 100), 1))
        # SSTV tone band guides: 1200 sync, 1500 black, 2300 white
        for hz, label in ((1200, None), (1500, None), (2300, None)):
            x = int(hz / 4000.0 * w)
            p.drawLine(x, 16, x, self.height() - 2)
        p.end()

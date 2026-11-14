"""Azimuth/elevation polar sky map."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class PolarSkyMap(QWidget):
    """Draws the satellite track across the sky for one pass.

    North is up, east is right; the rim is 0 deg elevation and the centre
    is the zenith (90 deg).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self._path: List[Tuple[float, float]] = []   # (az, el) along the pass
        self._position: Optional[Tuple[float, float]] = None
        self._progress: float = 0.0                  # 0..1 along the path
        self._below_horizon = True
        self._pass_label = ""

    # ------------------------------------------------------------------
    def set_pass(self, path: List[Tuple[float, float]], label: str = "") -> None:
        self._path = list(path)
        self._pass_label = label
        self.update()

    def set_position(self, az: float, el: float, progress: float = -1.0) -> None:
        self._position = (az, el)
        self._below_horizon = el <= 0.0
        if progress >= 0.0:
            self._progress = min(max(progress, 0.0), 1.0)
        self.update()

    # ------------------------------------------------------------------
    def _az_el_to_xy(self, az: float, el: float, rect: QRectF) -> QPointF:
        r_max = min(rect.width(), rect.height()) / 2.0 - 18.0
        r = (90.0 - max(min(el, 90.0), 0.0)) / 90.0 * r_max
        a = math.radians(az)
        cx = rect.center().x()
        cy = rect.center().y()
        return QPointF(cx + r * math.sin(a), cy - r * math.cos(a))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        p.fillRect(rect, QColor(10, 14, 22))

        pen = QPen(QColor(70, 90, 120), 1)
        p.setPen(pen)
        centre = rect.center()
        r_max = min(rect.width(), rect.height()) / 2.0 - 18.0

        # elevation rings
        for el in (0, 30, 60):
            r = (90 - el) / 90.0 * r_max
            p.drawEllipse(centre, r, r)
        # azimuth ticks + cardinal labels
        font = QFont(p.font())
        font.setPointSize(8)
        p.setFont(font)
        for az in range(0, 360, 30):
            a = math.radians(az)
            x1 = centre.x() + (r_max + 3) * math.sin(a)
            y1 = centre.y() - (r_max + 3) * math.cos(a)
            x2 = centre.x() + (r_max + 8) * math.sin(a)
            y2 = centre.y() - (r_max + 8) * math.cos(a)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            lx = centre.x() + (r_max + 14) * math.sin(a)
            ly = centre.y() - (r_max + 14) * math.cos(a)
            name = {0: "N", 90: "E", 180: "S", 270: "W"}.get(az)
            if name:
                p.drawText(QRectF(lx - 10, ly - 8, 20, 16),
                           Qt.AlignCenter, name)
            else:
                p.drawText(QRectF(lx - 10, ly - 8, 20, 16),
                           Qt.AlignCenter, str(az))

        # ground station
        p.setPen(QPen(QColor(120, 140, 170), 1))
        p.drawPoint(centre)

        # pass path
        if self._path:
            pts_future = []
            pts_past = []
            n = len(self._path)
            for i, (az, el) in enumerate(self._path):
                if el < 0:
                    continue
                pt = self._az_el_to_xy(az, el, rect)
                (pts_past if (i / max(n - 1, 1)) <= self._progress
                 else pts_future).append(pt)
            pen_dim = QPen(QColor(90, 120, 180, 140), 2, Qt.DashLine)
            p.setPen(pen_dim)
            for a, b in zip(pts_future, pts_future[1:]):
                p.drawLine(a, b)
            pen_past = QPen(QColor(120, 200, 255), 2)
            p.setPen(pen_past)
            for a, b in zip(pts_past, pts_past[1:]):
                p.drawLine(a, b)
            if pts_future:
                p.setPen(QPen(QColor(90, 120, 180), 1))
                p.setBrush(QColor(90, 120, 180))
                p.drawEllipse(pts_future[0], 3, 3)   # AOS marker
            if pts_past:
                p.setPen(QPen(QColor(120, 200, 255), 1))
                p.setBrush(QColor(120, 200, 255))
                p.drawEllipse(pts_past[-1], 3, 3)    # current/LOS marker

        # current position
        if self._position is not None and not self._below_horizon:
            az, el = self._position
            pt = self._az_el_to_xy(az, el, rect)
            p.setPen(QPen(QColor(255, 210, 80), 2))
            p.setBrush(QColor(255, 210, 80))
            p.drawEllipse(pt, 5, 5)
            p.setPen(QColor(255, 230, 150))
            p.drawText(QRectF(pt.x() + 8, pt.y() - 8, 60, 16),
                       Qt.AlignLeft, "ISS")

        # label
        if self._pass_label:
            p.setPen(QColor(160, 180, 210))
            p.drawText(QRectF(rect.left() + 6, rect.top() + 4,
                              rect.width() - 12, 18),
                       Qt.AlignLeft, self._pass_label)
        if self._position is not None and self._below_horizon:
            p.setPen(QColor(130, 150, 180))
            p.drawText(rect, Qt.AlignHCenter | Qt.AlignBottom,
                       "ISS below horizon")
        p.end()

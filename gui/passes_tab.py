"""Pass predictions tab: ground station setup and the pass list."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QHBoxLayout as QHBox,
                               QHeaderView, QLabel, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from core.tracker import PassInfo

LOCAL_TZ = datetime.now(timezone.utc).astimezone().tzinfo


class PassesTab(QWidget):
    stationChanged = Signal(float, float, float, float)  # lat, lon, alt, min_el
    predictRequested = Signal(float)                      # hours
    refreshTleRequested = Signal()
    armRequested = Signal(object)                         # PassInfo
    autoRecordToggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._passes: List[PassInfo] = []
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        station = QGroupBox("Ground station")
        form = QFormLayout(station)
        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(4)
        self.lat_spin.setSuffix(" deg")
        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(4)
        self.lon_spin.setSuffix(" deg")
        self.alt_spin = QDoubleSpinBox()
        self.alt_spin.setRange(-500.0, 9000.0)
        self.alt_spin.setSuffix(" m")
        self.min_el_spin = QDoubleSpinBox()
        self.min_el_spin.setRange(0.0, 60.0)
        self.min_el_spin.setSuffix(" deg")
        self.min_el_spin.setToolTip("Only show passes reaching at least this "
                                    "maximum elevation")
        form.addRow("Latitude", self.lat_spin)
        form.addRow("Longitude", self.lon_spin)
        form.addRow("Altitude", self.alt_spin)
        form.addRow("Min. max-elevation", self.min_el_spin)
        self.apply_btn = QPushButton("Apply station && predict")
        self.apply_btn.clicked.connect(self._on_apply)
        form.addRow(self.apply_btn)
        root.addWidget(station)

        tle_row = QHBox()
        self.tle_label = QLabel("TLE: -")
        self.tle_refresh_btn = QPushButton("Refresh TLE now")
        self.tle_refresh_btn.clicked.connect(self.refreshTleRequested.emit)
        tle_row.addWidget(self.tle_label, 1)
        tle_row.addWidget(self.tle_refresh_btn)
        root.addLayout(tle_row)

        pred_row = QHBox()
        pred_row.addWidget(QLabel("Predict next"))
        self.hours_spin = QDoubleSpinBox()
        self.hours_spin.setRange(6.0, 24 * 14.0)
        self.hours_spin.setDecimals(0)
        self.hours_spin.setSuffix(" h")
        self.hours_spin.setValue(96.0)
        self.predict_btn = QPushButton("Predict passes")
        self.predict_btn.clicked.connect(
            lambda: self.predictRequested.emit(self.hours_spin.value()))
        self.predict_btn.setEnabled(False)
        pred_row.addWidget(self.hours_spin)
        pred_row.addWidget(self.predict_btn)
        pred_row.addStretch(1)
        self.auto_check = QCheckBox("Auto-record armed pass")
        self.auto_check.setToolTip("Start recording automatically at AOS "
                                   "(minus pre-roll) and stop after LOS")
        self.auto_check.stateChanged.connect(
            lambda s: self.autoRecordToggled.emit(s == Qt.Checked))
        pred_row.addWidget(self.auto_check)
        root.addLayout(pred_row)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["AOS (local)", "Duration", "Max el", "AOS az", "LOS az", "Countdown"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table, 1)

        action_row = QHBox()
        self.arm_btn = QPushButton("Track && record selected pass")
        self.arm_btn.setEnabled(False)
        self.arm_btn.clicked.connect(self._on_arm)
        action_row.addStretch(1)
        action_row.addWidget(self.arm_btn)
        root.addLayout(action_row)

    # ------------------------------------------------------------------
    def set_station(self, lat: float, lon: float, alt: float, min_el: float,
                    auto_record: bool) -> None:
        self.lat_spin.setValue(lat)
        self.lon_spin.setValue(lon)
        self.alt_spin.setValue(alt)
        self.min_el_spin.setValue(min_el)
        self.auto_check.setChecked(auto_record)

    def set_tle_status(self, text: str) -> None:
        self.tle_label.setText(text)

    def set_tracker_ready(self, ready: bool) -> None:
        self.predict_btn.setEnabled(ready)

    # ------------------------------------------------------------------
    def _on_apply(self) -> None:
        self.stationChanged.emit(self.lat_spin.value(), self.lon_spin.value(),
                                 self.alt_spin.value(), self.min_el_spin.value())

    def _on_arm(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._passes):
            self.armRequested.emit(self._passes[row])

    # ------------------------------------------------------------------
    def set_passes(self, passes: List[PassInfo]) -> None:
        self._passes = list(passes)
        self.table.setRowCount(len(passes))
        for i, p in enumerate(passes):
            aos_local = p.aos.astimezone(LOCAL_TZ).strftime("%a %Y-%m-%d %H:%M:%S")
            items = [
                aos_local,
                f"{int(p.duration_s // 60)}:{int(p.duration_s % 60):02d}",
                f"{p.max_el_deg:.1f} deg",
                f"{p.aos_az_deg:.0f} deg",
                f"{p.los_az_deg:.0f} deg",
                "",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, item)
        self.arm_btn.setEnabled(len(passes) > 0)
        self._armed_row = -1

    def set_armed(self, pass_info: Optional[PassInfo]) -> None:
        self._armed_row = -1
        if pass_info is not None:
            for i, p in enumerate(self._passes):
                if p.aos == pass_info.aos:
                    self._armed_row = i
                    break
        self._update_countdowns()

    _armed_row = -1

    def _update_countdowns(self) -> None:
        now = datetime.now(timezone.utc)
        for i, p in enumerate(self._passes):
            item = self.table.item(i, 5)
            if item is None:
                continue
            if now < p.aos:
                d = p.aos - now
                item.setText(f"in {int(d.total_seconds() // 60):02d}:"
                             f"{int(d.total_seconds() % 60):02d}")
            elif now < p.los:
                item.setText("IN PROGRESS")
            else:
                item.setText("past")
            bold = (i == self._armed_row)
            font = item.font()
            font.setBold(bold)
            item.setFont(font)
            item.setForeground(Qt.green if bold else Qt.white)

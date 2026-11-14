"""Live pass tab: sky map, readouts, waterfall, record controls, decode."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from core.tracker import PassInfo
from gui.decode_panel import DecodePanel
from gui.polar_map import PolarSkyMap
from gui.waterfall import Waterfall


class LiveTab(QWidget):
    startRecordingRequested = Signal()
    stopRecordingRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # left: sky map
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.map = PolarSkyMap()
        self.pass_label = QLabel("No pass armed")
        self.pass_label.setAlignment(Qt.AlignCenter)
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        lv.addWidget(self.map, 1)
        lv.addWidget(self.pass_label)
        lv.addWidget(self.countdown_label)

        # right: readouts + waterfall + controls
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        readouts = QGroupBox("Tracking")
        form = QFormLayout(readouts)
        self.utc_label = self._ro()
        self.az_label = self._ro()
        self.el_label = self._ro()
        self.range_label = self._ro()
        self.velocity_label = self._ro()
        self.doppler_label = self._ro()
        self.tuned_label = self._ro()
        self.rec_label = self._ro()
        form.addRow("UTC", self.utc_label)
        form.addRow("Azimuth", self.az_label)
        form.addRow("Elevation", self.el_label)
        form.addRow("Range", self.range_label)
        form.addRow("Range rate", self.velocity_label)
        form.addRow("Doppler", self.doppler_label)
        form.addRow("Signal freq", self.tuned_label)
        form.addRow("Recording", self.rec_label)
        rv.addWidget(readouts)

        self.waterfall = Waterfall()
        rv.addWidget(self.waterfall, 1)

        controls = QHBoxLayout()
        self.record_btn = QPushButton("Start recording")
        self.record_btn.clicked.connect(self.startRecordingRequested.emit)
        self.stop_record_btn = QPushButton("Stop")
        self.stop_record_btn.setEnabled(False)
        self.stop_record_btn.clicked.connect(self.stopRecordingRequested.emit)
        self.status_label = QLabel("idle")
        controls.addWidget(self.record_btn)
        controls.addWidget(self.stop_record_btn)
        controls.addStretch(1)
        controls.addWidget(self.status_label)
        rv.addLayout(controls)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self.decode_panel = DecodePanel()
        root.addWidget(self.decode_panel)

    @staticmethod
    def _ro() -> QLabel:
        lbl = QLabel("-")
        lbl.setStyleSheet("font-family: monospace;")
        return lbl

    # ------------------------------------------------------------------
    def set_pass(self, pass_info: Optional[PassInfo],
                 path=None, label: str = "") -> None:
        if pass_info is None and not path:
            self.map.set_pass([], "")
            self.pass_label.setText("No pass armed")
            return
        self.map.set_pass(path or [], label or self.pass_label.text())
        if pass_info is not None:
            self.pass_label.setText(
                f"{label} AOS {pass_info.aos.strftime('%H:%M:%SZ')} - "
                f"LOS {pass_info.los.strftime('%H:%M:%SZ')} UTC "
                f"(max {pass_info.max_el_deg:.0f} deg)")

    def update_position(self, az: float, el: float, range_km: float,
                        range_rate: float, doppler_hz: float,
                        nominal_freq_hz: float, progress: float) -> None:
        self.utc_label.setText(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        self.az_label.setText(f"{az:6.1f} deg")
        self.el_label.setText(f"{el:6.1f} deg" if el > 0 else "below horizon")
        self.range_label.setText(f"{range_km:8.1f} km")
        self.velocity_label.setText(f"{range_rate:+6.2f} km/s")
        self.doppler_label.setText(f"{doppler_hz:+8.0f} Hz")
        tuned = (nominal_freq_hz + doppler_hz) / 1e6
        self.tuned_label.setText(f"{tuned:.6f} MHz")
        self.map.set_position(az, el, progress)

    def set_countdown(self, text: str) -> None:
        self.countdown_label.setText(text)

    def set_recording(self, active: bool) -> None:
        self.record_btn.setEnabled(not active)
        self.stop_record_btn.setEnabled(active)
        if active:
            self.rec_label.setText("RECORDING")
            self.rec_label.setStyleSheet("color: #ff5555; font-weight: bold;")
        else:
            self.rec_label.setText("idle")
            self.rec_label.setStyleSheet("")

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    # pipeline callbacks -------------------------------------------------
    def push_spectrum(self, row) -> None:
        self.waterfall.push_row(row)

    def handle_sstv_event(self, ev) -> None:
        self.decode_panel.handle_event(ev)

    def pass_finished(self) -> None:
        self.decode_panel.clear_finished_pass()

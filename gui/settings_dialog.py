"""Settings dialog."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from core.settings import Settings
from core.sources.sdr import probe_rtlsdr


class SettingsDialog(QDialog):
    def __init__(self, s: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.s = s
        tabs = QTabWidget()

        # --- radio tab ---
        radio = QWidget()
        rform = QFormLayout(radio)
        self.nominal = self._spin(1.0, 30000.0, 6, s.nominal_freq_hz / 1e6)
        self.nominal.setSuffix(" MHz")
        rform.addRow("Nominal downlink", self.nominal)
        self.offset = self._spin(0.0, 100000.0, 0, s.tuning_offset_hz)
        self.offset.setSuffix(" Hz")
        self.offset.setToolTip("The SDR is tuned this far below the nominal "
                               "frequency to avoid its DC spike")
        rform.addRow("Tuning offset", self.offset)
        self.fine = self._spin(-5000.0, 5000.0, 0, s.fine_offset_hz)
        self.fine.setSuffix(" Hz")
        self.fine.setToolTip("Residual frequency error correction (e.g. "
                             "fractional ppm at the downlink frequency)")
        rform.addRow("Fine offset", self.fine)
        self.ppm = self._spin(-200.0, 200.0, 0, s.ppm)
        self.ppm.setSuffix(" ppm")
        rform.addRow("SDR frequency error", self.ppm)
        self.gain_auto = QCheckBox("Automatic gain")
        self.gain_auto.setChecked(s.gain_auto)
        rform.addRow(self.gain_auto)
        self.gain_db = self._spin(-10.0, 60.0, 1, s.manual_gain_db)
        self.gain_db.setSuffix(" dB")
        rform.addRow("Manual gain", self.gain_db)
        self.chan_bw = self._spin(4000.0, 60000.0, 0, s.channel_bw_hz)
        self.chan_bw.setSuffix(" Hz")
        rform.addRow("Channel bandwidth", self.chan_bw)
        self.deemph = QCheckBox("Apply 750 us de-emphasis")
        self.deemph.setChecked(s.deemphasis)
        rform.addRow(self.deemph)
        self.doppler = QCheckBox("Software doppler correction")
        self.doppler.setChecked(s.doppler_correction)
        rform.addRow(self.doppler)
        self.monitor = QCheckBox("Monitor audio on speaker")
        self.monitor.setChecked(s.monitor_audio)
        rform.addRow(self.monitor)
        tabs.addTab(radio, "Radio")

        # --- passes tab ---
        passes = QWidget()
        pform = QFormLayout(passes)
        self.preroll = self._spin(0.0, 300.0, 0, s.preroll_s)
        self.preroll.setSuffix(" s")
        pform.addRow("Pre-roll before AOS", self.preroll)
        self.postroll = self._spin(0.0, 300.0, 0, s.postroll_s)
        self.postroll.setSuffix(" s")
        pform.addRow("Post-roll after LOS", self.postroll)
        tabs.addTab(passes, "Recording")

        # --- source tab ---
        source = QWidget()
        sform = QFormLayout(source)
        self.source = QComboBox()
        self.source.addItem("RTL-SDR dongle", "rtlsdr")
        self.source.addItem("Sound card (line-in/mic)", "soundcard")
        self.source.addItem("Simulated pass (no hardware)", "sim")
        self.source.setCurrentIndex(
            ["rtlsdr", "soundcard", "sim"].index(s.source_type))
        sform.addRow("Signal source", self.source)
        self.sim_snr = self._spin(0.0, 60.0, 0, s.sim_snr_db)
        self.sim_snr.setSuffix(" dB")
        sform.addRow("Simulator SNR", self.sim_snr)

        check_row = QHBoxLayout()
        self.rtlsdr_check_btn = QPushButton("Check RTL-SDR connection")
        self.rtlsdr_check_btn.setToolTip(
            "Verify that pyrtlsdr and the librtlsdr driver are installed and "
            "that a dongle is reachable")
        self.rtlsdr_check_btn.clicked.connect(self._check_rtlsdr)
        self.rtlsdr_check_label = QLabel("")
        self.rtlsdr_check_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        check_row.addWidget(self.rtlsdr_check_btn)
        check_row.addWidget(self.rtlsdr_check_label, 1)
        sform.addRow(check_row)
        tabs.addTab(source, "Source")

        # --- TLE tab ---
        tle = QWidget()
        tform = QFormLayout(tle)
        self.tle_url = self._text(s.tle_url)
        tform.addRow("TLE URL", self.tle_url)
        self.tle_age = self._spin(0.5, 24 * 30.0, 1, s.tle_max_age_hours)
        self.tle_age.setSuffix(" h")
        tform.addRow("Cache max age", self.tle_age)
        tabs.addTab(tle, "TLE")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _check_rtlsdr(self) -> None:
        """Probe the RTL-SDR stack and report the result to the user."""
        from PySide6.QtWidgets import QApplication
        self.rtlsdr_check_btn.setEnabled(False)
        self.rtlsdr_check_label.setText("checking...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            ok, message = probe_rtlsdr()
        finally:
            QApplication.restoreOverrideCursor()
            self.rtlsdr_check_btn.setEnabled(True)
        self._show_rtlsdr_result(ok, message)

    def _show_rtlsdr_result(self, ok: bool, message: str) -> None:
        self.rtlsdr_check_label.setText(message.splitlines()[0])
        if ok:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("RTL-SDR check")
            box.setText("The RTL-SDR dongle is ready to use.")
            box.setInformativeText(message)
            box.exec()
        else:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("RTL-SDR not available")
            box.setText("The RTL-SDR check failed.")
            box.setInformativeText(message)
            box.setTextFormat(Qt.PlainText)  # keep the setup steps readable
            box.exec()

    @staticmethod
    def _spin(lo, hi, dec, val):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(dec)
        sb.setValue(val)
        return sb

    @staticmethod
    def _text(val):
        from PySide6.QtWidgets import QLineEdit
        return QLineEdit(str(val))

    def apply(self) -> Settings:
        s = self.s
        s.nominal_freq_hz = self.nominal.value() * 1e6
        s.tuning_offset_hz = self.offset.value()
        s.fine_offset_hz = self.fine.value()
        s.ppm = int(round(self.ppm.value()))
        s.gain_auto = self.gain_auto.isChecked()
        s.manual_gain_db = self.gain_db.value()
        s.channel_bw_hz = self.chan_bw.value()
        s.deemphasis = self.deemph.isChecked()
        s.doppler_correction = self.doppler.isChecked()
        s.monitor_audio = self.monitor.isChecked()
        s.preroll_s = self.preroll.value()
        s.postroll_s = self.postroll.value()
        s.source_type = self.source.currentData()
        s.sim_snr_db = self.sim_snr.value()
        s.tle_url = self.tle_url.text().strip()
        s.tle_max_age_hours = self.tle_age.value()
        return s

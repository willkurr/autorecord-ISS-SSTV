"""Main application window: tabs, services, scheduling, live tracking."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QLabel, QMainWindow, QMessageBox, QStatusBar,
                               QTabWidget)

from core.images import ImageStore
from core.pipeline import PassPipeline
from core.scheduler import PassScheduler
from core.settings import Settings, load_settings, save_settings
from core.tle import ISS_CATALOG_NUMBER, TLEProvider
from core.tracker import PassInfo, Tracker
from gui.images_tab import ImagesTab
from gui.live_tab import LiveTab
from gui.passes_tab import PassesTab
from gui.settings_dialog import SettingsDialog
from gui.workers import JobRunner
from util import APP_NAME, APP_VERSION

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, simulate: bool = False):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1280, 860)

        self.settings: Settings = load_settings()
        if simulate:
            self.settings.source_type = "sim"
        self.tle_provider = TLEProvider(url=self.settings.tle_url,
                                        max_age_hours=self.settings.tle_max_age_hours)
        self.tracker: Optional[Tracker] = None
        self.image_store = ImageStore()
        self.pipeline: Optional[PassPipeline] = None
        self.armed_pass: Optional[PassInfo] = None
        self.pass_path = []
        self._tle_runner = JobRunner()
        self._tle_runner.succeeded.connect(self._on_tle_loaded)
        self._tle_runner.failed.connect(self._on_tle_failed)
        self._predict_runner = JobRunner()
        self._predict_runner.succeeded.connect(self._on_predicted)
        self._predict_runner.failed.connect(self._on_predict_failed)

        self._build_ui()
        self._build_menu()
        self._start_timers()
        self.refresh_tle(background=True)

    # ------------------------------------------------------------------
    # UI construction
    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.passes_tab = PassesTab()
        self.live_tab = LiveTab()
        self.images_tab = ImagesTab(self.image_store)
        self.tabs.addTab(self.passes_tab, "Passes")
        self.tabs.addTab(self.live_tab, "Live pass")
        self.tabs.addTab(self.images_tab, "Images")
        self.setCentralWidget(self.tabs)

        self.passes_tab.set_station(
            self.settings.lat_deg, self.settings.lon_deg, self.settings.alt_m,
            self.settings.min_el_deg, self.settings.auto_record)
        self.passes_tab.stationChanged.connect(self._on_station_changed)
        self.passes_tab.predictRequested.connect(self._predict_passes)
        self.passes_tab.refreshTleRequested.connect(
            lambda: self.refresh_tle(background=True, force=True))
        self.passes_tab.armRequested.connect(self.arm_pass)
        self.passes_tab.autoRecordToggled.connect(self._on_auto_record)

        self.live_tab.startRecordingRequested.connect(
            lambda: self.scheduler.start_recording(self.armed_pass))
        self.live_tab.stopRecordingRequested.connect(self.scheduler_stop)

        self.scheduler = PassScheduler(self.settings, self._pipeline_factory)
        self.scheduler.recordingStarted.connect(self._on_recording_started)
        self.scheduler.recordingStopped.connect(self._on_recording_stopped)
        self.scheduler.statusChanged.connect(self.live_tab.set_status)
        self.scheduler.set_auto_record(self.settings.auto_record)

        self.statusBar().showMessage("Ready")

    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self._open_settings)
        m_file.addAction(settings_action)
        m_file.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.triggered.connect(self.close)
        m_file.addAction(quit_action)

        m_pass = self.menuBar().addMenu("&Pass")
        self.action_record = QAction("Start &recording now", self)
        self.action_record.triggered.connect(
            lambda: self.scheduler.start_recording(self.armed_pass))
        m_pass.addAction(self.action_record)
        self.action_stop = QAction("&Stop recording", self)
        self.action_stop.triggered.connect(self.scheduler_stop)
        m_pass.addAction(self.action_stop)

        m_help = self.menuBar().addMenu("&Help")
        about_action = QAction(f"&About {APP_NAME}", self)
        about_action.triggered.connect(self._about)
        m_help.addAction(about_action)

    def _start_timers(self) -> None:
        self.track_timer = QTimer(self)
        self.track_timer.setInterval(250)
        self.track_timer.timeout.connect(self._update_tracking)
        self.track_timer.start()
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(1000)
        self.ui_timer.timeout.connect(self._update_countdowns)
        self.ui_timer.start()

    # ------------------------------------------------------------------
    # TLE / tracker
    def refresh_tle(self, background: bool = True, force: bool = False) -> None:
        self.passes_tab.set_tle_status("TLE: updating...")
        if not background:
            self._load_tle(force)
            return

        def job():
            self.tle_provider.update(force=force)
            return self.tle_provider.get(ISS_CATALOG_NUMBER)

        self._tle_runner.run(job)

    def _load_tle(self, force: bool) -> None:
        try:
            self.tle_provider.update(force=force)
            self._on_tle_loaded(self.tle_provider.get(ISS_CATALOG_NUMBER))
        except Exception as exc:  # noqa: BLE001
            self._on_tle_failed(str(exc))

    def _on_tle_loaded(self, tle) -> None:
        self.tracker = Tracker(tle)
        self.tracker.set_station(self.settings.lat_deg, self.settings.lon_deg,
                                 self.settings.alt_m)
        epoch = f"epoch {tle.line1[18:32].strip()}"
        self.passes_tab.set_tle_status(
            f"TLE: {tle.name} loaded ({epoch}, {self.tle_provider.cache_age_text()})")
        self.passes_tab.set_tracker_ready(True)
        self.statusBar().showMessage(f"Tracking {tle.name} from fresh elements", 5000)
        self._predict_passes(self.settings.predict_hours)

    def _on_tle_failed(self, message: str) -> None:
        self.passes_tab.set_tle_status(f"TLE: unavailable ({message})")
        QMessageBox.warning(self, "TLE update failed",
                            f"Could not obtain Keplerian elements:\n{message}\n\n"
                            "Pass prediction is unavailable until TLEs load.")

    # ------------------------------------------------------------------
    # station + prediction
    def _on_station_changed(self, lat: float, lon: float, alt: float,
                            min_el: float) -> None:
        self.settings.lat_deg = lat
        self.settings.lon_deg = lon
        self.settings.alt_m = alt
        self.settings.min_el_deg = min_el
        save_settings(self.settings)
        if self.tracker is not None:
            self.tracker.set_station(lat, lon, alt)
            self._predict_passes(self.settings.predict_hours)

    def _predict_passes(self, hours: float) -> None:
        if self.tracker is None:
            return
        self.settings.predict_hours = hours
        self.passes_tab.predict_btn.setEnabled(False)
        self.passes_tab.predict_btn.setText("Predicting...")
        tracker = self.tracker
        lat, lon, alt = (self.settings.lat_deg, self.settings.lon_deg,
                         self.settings.alt_m)
        min_el = self.settings.min_el_deg
        now = datetime.now(timezone.utc)

        def job():
            return tracker.next_passes(now, hours=hours, min_el_deg=min_el)

        self._predict_runner.run(job)

    def _on_predicted(self, passes) -> None:
        self.passes_tab.predict_btn.setEnabled(True)
        self.passes_tab.predict_btn.setText("Predict passes")
        self.passes_tab.set_passes(passes)
        if self.armed_pass is not None:
            self.passes_tab.set_armed(self.armed_pass)
        # arm the next upcoming pass automatically if none armed
        if self.armed_pass is None and passes:
            self.arm_pass(passes[0])

    def _on_predict_failed(self, message: str) -> None:
        self.passes_tab.predict_btn.setEnabled(True)
        self.passes_tab.predict_btn.setText("Predict passes")
        QMessageBox.warning(self, "Prediction failed", message)

    def _on_auto_record(self, enabled: bool) -> None:
        self.settings.auto_record = enabled
        save_settings(self.settings)
        self.scheduler.set_auto_record(enabled)

    # ------------------------------------------------------------------
    # pass arming
    def arm_pass(self, pass_info: Optional[PassInfo]) -> None:
        self.armed_pass = pass_info
        self.passes_tab.set_armed(pass_info)
        if pass_info is None or self.tracker is None:
            self.live_tab.set_pass(None)
            return
        azs, els = self.tracker.pass_path(pass_info, step_s=10.0)
        self.pass_path = list(zip(azs, els))
        self.live_tab.set_pass(pass_info, self.pass_path,
                               label=f"ISS pass {pass_info.aos.strftime('%Y-%m-%d')}")
        self.statusBar().showMessage(
            f"Armed pass: AOS {pass_info.aos.strftime('%H:%M:%SZ')} UTC, "
            f"max {pass_info.max_el_deg:.0f} deg", 8000)

    # ------------------------------------------------------------------
    # live tracking
    def _update_tracking(self) -> None:
        if self.tracker is None:
            return
        now = datetime.now(timezone.utc)
        pos = self.tracker.position_at(now)
        doppler = pos.doppler_hz(self.settings.nominal_freq_hz)
        progress = -1.0
        if self.armed_pass is not None:
            span = self.armed_pass.duration_s
            progress = (now - self.armed_pass.aos).total_seconds() / max(span, 1)
        self.live_tab.update_position(
            pos.az_deg, pos.el_deg, pos.range_km, pos.range_rate_km_s,
            doppler, self.settings.nominal_freq_hz, progress)

    def _update_countdowns(self) -> None:
        self.passes_tab._update_countdowns()
        now = datetime.now(timezone.utc)
        if self.armed_pass is None:
            self.live_tab.set_countdown("")
            return
        p = self.armed_pass
        if now < p.aos:
            d = p.aos - now
            self.live_tab.set_countdown(
                f"Next armed pass in {int(d.total_seconds() // 60):02d}:"
                f"{int(d.total_seconds() % 60):02d} (AOS {p.aos.strftime('%H:%M:%SZ')} UTC)")
        elif now < p.los:
            left = p.los - now
            self.live_tab.set_countdown(
                f"Pass in progress - LOS in {int(left.total_seconds() // 60):02d}:"
                f"{int(left.total_seconds() % 60):02d}")

    # ------------------------------------------------------------------
    # recording
    def _pipeline_factory(self, pass_info: Optional[PassInfo]) -> PassPipeline:
        pipeline = PassPipeline(self.settings, self.tracker, pass_info,
                                self.image_store)
        pipeline.sstvEvent.connect(self.live_tab.handle_sstv_event)
        pipeline.spectrumRow.connect(self.live_tab.push_spectrum)
        return pipeline

    def scheduler_stop(self) -> None:
        self.scheduler.stop_recording()

    def _on_recording_started(self, pass_info) -> None:
        self.live_tab.set_recording(True)
        self.action_record.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.tabs.setCurrentWidget(self.live_tab)

    def _on_recording_stopped(self, result) -> None:
        self.live_tab.set_recording(False)
        self.action_record.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.live_tab.pass_finished()
        self.images_tab.refresh()
        wav = result.get("wav_path")
        msg = f"Recording saved: {wav.name if wav else 'n/a'} " \
              f"({result.get('images', 0)} image(s))"
        self.statusBar().showMessage(msg, 10000)

    # ------------------------------------------------------------------
    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings = dlg.apply()
            save_settings(self.settings)
            self.tle_provider.url = self.settings.tle_url
            self.tle_provider.max_age_hours = self.settings.tle_max_age_hours
            self.statusBar().showMessage("Settings saved", 5000)

    def _about(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{APP_VERSION}<br><br>"
            "ISS slow-scan television tracking, recording and decoding.<br><br>"
            "Elements courtesy of AMSAT (amsat.org).<br>"
            "Robot 36 decoding per the published SSTV specification.")

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        if self.scheduler.pipeline is not None:
            if QMessageBox.question(
                    self, "Recording in progress",
                    "A recording is in progress. Stop it and exit?") != QMessageBox.Yes:
                event.ignore()
                return
            self.scheduler.stop_recording()
        save_settings(self.settings)
        super().closeEvent(event)

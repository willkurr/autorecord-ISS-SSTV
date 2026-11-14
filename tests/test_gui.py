"""GUI smoke test: construction, prediction, arming, simulated recording."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from core.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    import core.images
    import core.pipeline
    import gui.main_window as mw
    import util

    monkeypatch.setattr(mw, "load_settings", lambda *a, **k: Settings())
    monkeypatch.setattr(mw, "save_settings", lambda s, *a, **k: None)
    monkeypatch.setattr("core.settings.SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(util, "DATA_DIR", tmp_path)
    monkeypatch.setattr(util, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(util, "RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr(core.images, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(core.pipeline, "RECORDINGS_DIR", tmp_path / "recordings")

    win = mw.MainWindow(simulate=True)
    win.settings.source_type = "sim"
    win.settings.sim_speed = 20.0
    win.show()
    yield win
    win.close()


def _wait_for_tracker(qapp, win, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline and win.tracker is None:
        qapp.processEvents()
        time.sleep(0.05)
    assert win.tracker is not None, "TLE/tracker never loaded"


def test_gui_constructs_and_predicts(qapp, window):
    _wait_for_tracker(qapp, window)
    # prediction auto-runs after TLE load
    deadline = time.time() + 60
    while time.time() < deadline and window.passes_tab.table.rowCount() == 0:
        qapp.processEvents()
        time.sleep(0.05)
    assert window.passes_tab.table.rowCount() > 0
    assert window.armed_pass is not None  # first pass auto-armed
    assert window.live_tab.map._path  # sky map has a track


def test_settings_dialog_rtlsdr_check(qapp, window, monkeypatch):
    """The Source tab check button reports missing-library instructions."""
    from gui.settings_dialog import SettingsDialog

    shown = []
    monkeypatch.setattr(
        "gui.settings_dialog.QMessageBox.exec",
        lambda self: shown.append((self.icon(), self.text(),
                                   self.informativeText())))
    monkeypatch.setitem(__import__("sys").modules, "rtlsdr", None)  # no pyrtlsdr

    dlg = SettingsDialog(window.settings, window)
    dlg.rtlsdr_check_btn.click()

    assert shown, "no result dialog was shown"
    icon, text, detail = shown[0]
    from PySide6.QtWidgets import QMessageBox
    assert icon == QMessageBox.Warning
    assert "failed" in text.lower()
    assert "brew install librtlsdr" in detail
    assert "Zadig" in detail
    assert "apt install librtlsdr-dev" in detail
    assert "checking" not in dlg.rtlsdr_check_label.text()
    dlg.close()


def test_settings_dialog_rtlsdr_check_success(qapp, window, monkeypatch):
    import sys
    import types
    from gui.settings_dialog import SettingsDialog

    module = types.ModuleType("rtlsdr")

    class FakeRtlSdr:
        device_name = "Generic RTL2832U OEM"

        def read_samples(self, n):
            return [0] * n

        def close(self):
            pass

    module.RtlSdr = FakeRtlSdr
    monkeypatch.setitem(sys.modules, "rtlsdr", module)

    shown = []
    monkeypatch.setattr(
        "gui.settings_dialog.QMessageBox.exec",
        lambda self: shown.append((self.icon(), self.text())))
    dlg = SettingsDialog(window.settings, window)
    dlg.rtlsdr_check_btn.click()
    from PySide6.QtWidgets import QMessageBox
    assert shown and shown[0][0] == QMessageBox.Information
    assert "ready to use" in shown[0][1]
    dlg.close()


def test_gui_simulated_recording_cycle(qapp, window, tmp_path):
    _wait_for_tracker(qapp, window)

    from core.tracker import PassInfo
    now = datetime.now(timezone.utc)
    passes = window.tracker.next_passes(now, hours=24.0, min_el_deg=20.0)
    assert passes
    p = passes[0]
    tca = p.aos + timedelta(seconds=p.duration_s / 2.0)
    # a short synthetic "pass" starting right now, around a real TCA so the
    # doppler profile is realistic; the sim is DSP-bound in the GUI, so keep
    # the window short - the image will decode partially, which still saves
    # to the browser
    fake = PassInfo(aos=now - timedelta(seconds=5),
                    los=now + timedelta(seconds=45),
                    max_el_deg=p.max_el_deg, aos_az_deg=p.aos_az_deg,
                    los_az_deg=p.los_az_deg)
    window.settings.preroll_s = 0.0
    window.settings.postroll_s = 0.0
    window.scheduler.arm_pass(fake)
    window.scheduler.set_auto_record(True)

    # drive the scheduler tick directly (QTimer may not fire under offscreen
    # pumping alone) and pump the event loop until recording completes
    stopped = False
    deadline = time.time() + 120
    while time.time() < deadline:
        qapp.processEvents()
        window.scheduler.tick()
        if window.scheduler.pipeline is None:
            wavs = list((tmp_path / "recordings").glob("*.wav"))
            if wavs:
                stopped = True
                break
        time.sleep(0.05)
    qapp.processEvents()
    assert stopped, "simulated recording cycle did not complete"
    assert window.images_tab.list_widget.count() >= 1, "decoded images not in browser"
    # a WAV recording exists with a metadata sidecar
    wavs = list((tmp_path / "recordings").glob("*.wav"))
    assert all(w.with_suffix(".json").exists() for w in wavs)

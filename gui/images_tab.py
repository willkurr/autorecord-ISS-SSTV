"""Image browser tab: gallery of decoded images + offline re-analysis."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QPushButton, QSplitter, QVBoxLayout,
                               QWidget, QPlainTextEdit)

from core.images import ImageStore
from util import RECORDINGS_DIR

log = logging.getLogger(__name__)


class OfflineDecodeWorker(QThread):
    """Decodes all images in a WAV file; emits each completed image."""

    progress = Signal(str)
    finished_decode = Signal(object)  # np.ndarray image
    done = Signal(int)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        from sstv.engine import decode_file
        from sstv.events import EventType
        count = 0
        self.progress.emit(f"Decoding {Path(self.path).name}...")

        def on_event(ev):
            if ev.type == EventType.LINE and ev.line % 24 == 0:
                self.progress.emit(
                    f"Decoding {Path(self.path).name}: line {ev.line}/240")
            elif ev.type == EventType.IMAGE_COMPLETE:
                nonlocal count
                count += 1
                self.finished_decode.emit(ev.image)

        try:
            decode_file(self.path, on_event=on_event)
        except Exception as exc:  # noqa: BLE001
            self.progress.emit(f"Decode failed: {exc}")
        self.done.emit(count)


class ImagesTab(QWidget):
    def __init__(self, store: ImageStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._worker: Optional[OfflineDecodeWorker] = None
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setIconSize(QSize(160, 120))
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setMovement(QListWidget.Static)
        self.list_widget.itemSelectionChanged.connect(self._show_detail)
        splitter.addWidget(self.list_widget)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.detail_image = QLabel()
        self.detail_image.setAlignment(Qt.AlignCenter)
        self.detail_image.setMinimumSize(320, 240)
        self.detail_image.setStyleSheet("background: #10141c;")
        rv.addWidget(self.detail_image, 1)
        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(120)
        rv.addWidget(self.detail_text)
        buttons = QHBoxLayout()
        self.export_btn = QPushButton("Export as...")
        self.export_btn.clicked.connect(self._export)
        self.folder_btn = QPushButton("Show in folder")
        self.folder_btn.clicked.connect(self._reveal)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete)
        self.reanalyze_btn = QPushButton("Re-analyze a recording...")
        self.reanalyze_btn.clicked.connect(self._reanalyze)
        buttons.addWidget(self.export_btn)
        buttons.addWidget(self.folder_btn)
        buttons.addWidget(self.delete_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.reanalyze_btn)
        rv.addLayout(buttons)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.list_widget.clear()
        for entry in reversed(self.store.entries()):
            path = self.store.image_path(entry)
            if not path.exists():
                continue
            icon = QIcon(str(path))
            item = QListWidgetItem(icon, entry.get("saved_utc", "")[:19])
            item.setData(Qt.UserRole, entry)
            self.list_widget.addItem(item)

    def current_entry(self):
        item = self.list_widget.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _show_detail(self) -> None:
        entry = self.current_entry()
        if entry is None:
            self.detail_image.clear()
            self.detail_text.clear()
            return
        path = self.store.image_path(entry)
        self.detail_image.setPixmap(
            QPixmap(str(path)).scaled(self.detail_image.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))
        mode = entry.get("mode") or "?"
        truncated = "yes" if entry.get("truncated") else "no"
        rec = entry.get("recording") or "-"
        freq = entry.get("frequency_hz")
        freq_txt = f"{freq / 1e6:.6f} MHz" if freq else "-"
        self.detail_text.setPlainText(
            f"file:   {path}\n"
            f"mode:   {mode}   truncated: {truncated}   "
            f"sync: {float(entry.get('sync_quality', 1.0)) * 100:.0f}%\n"
            f"source: {entry.get('source') or '-'}   "
            f"satellite: {entry.get('satellite') or '-'}   freq: {freq_txt}\n"
            f"saved:  {entry.get('saved_utc')}\n"
            f"from recording: {rec}")

    # ------------------------------------------------------------------
    def _export(self) -> None:
        entry = self.current_entry()
        if entry is None:
            return
        src = self.store.image_path(entry)
        dest, _ = QFileDialog.getSaveFileName(self, "Export image",
                                              str(Path.home() / src.name),
                                              "PNG images (*.png)")
        if dest:
            try:
                Path(dest).write_bytes(src.read_bytes())
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def _reveal(self) -> None:
        entry = self.current_entry()
        if entry is None:
            return
        path = self.store.image_path(entry)
        try:
            import subprocess
            import sys
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.information(self, "Image location", str(path))

    def _delete(self) -> None:
        entry = self.current_entry()
        if entry is None:
            return
        if QMessageBox.question(self, "Delete image",
                                f"Delete {entry['file']}?") == QMessageBox.Yes:
            self.store.delete(entry)
            self.refresh()

    # ------------------------------------------------------------------
    def _reanalyze(self) -> None:
        if self._worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a pass recording", str(RECORDINGS_DIR),
            "WAV audio (*.wav)")
        if not path:
            return
        self._worker = OfflineDecodeWorker(path)
        self._worker.progress.connect(lambda msg: self.detail_text.setPlainText(msg))
        self._worker.finished_decode.connect(self._save_decoded)
        self._worker.done.connect(self._decode_done)
        self._worker.start()

    def _save_decoded(self, image) -> None:
        try:
            self.store.save(image, meta={"mode": "Robot 36",
                                         "source": "re-analysis",
                                         "recording": None})
        except Exception:  # noqa: BLE001
            log.exception("saving re-analyzed image failed")

    def _decode_done(self, count: int) -> None:
        self.detail_text.setPlainText(f"Re-analysis complete: {count} image(s) decoded.")
        self._worker = None
        self.refresh()

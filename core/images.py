"""Image storage: saved PNGs plus a JSON index with metadata."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from util import IMAGES_DIR

log = logging.getLogger(__name__)

INDEX_NAME = "index.json"


class ImageStore:
    """Persists decoded images and their metadata under a directory."""

    def __init__(self, directory: Path = IMAGES_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / INDEX_NAME

    # ------------------------------------------------------------------
    def _load_index(self) -> List[Dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as exc:  # noqa: BLE001
            log.warning("Image index unreadable (%s); starting fresh", exc)
            return []

    def _save_index(self, entries: List[Dict[str, Any]]) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.index_path)

    # ------------------------------------------------------------------
    def save(self, image: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> Path:
        """Save a decoded image as PNG; returns the file path."""
        from PySide6.QtGui import QImage
        h, w, _ = image.shape
        qimg = QImage(np.ascontiguousarray(image).data, w, h, 3 * w,
                      QImage.Format_RGB888)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        n = len(self._load_index())
        path = self.directory / f"{stamp}_{n:04d}.png"
        i = 0
        while path.exists():
            i += 1
            path = self.directory / f"{stamp}_{n:04d}_{i}.png"
        if not qimg.save(str(path)):
            raise RuntimeError(f"failed to save image to {path}")
        entry = {
            "file": path.name,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "mode": (meta or {}).get("mode"),
            "truncated": (meta or {}).get("truncated", False),
            "sync_quality": (meta or {}).get("sync_quality", 1.0),
            "recording": (meta or {}).get("recording"),
            "pass_aos": (meta or {}).get("pass_aos"),
            "satellite": (meta or {}).get("satellite"),
            "frequency_hz": (meta or {}).get("frequency_hz"),
            "source": (meta or {}).get("source"),
        }
        entries = self._load_index()
        entries.append(entry)
        self._save_index(entries)
        log.info("Saved decoded image %s", path.name)
        return path

    # ------------------------------------------------------------------
    def entries(self) -> List[Dict[str, Any]]:
        return self._load_index()

    def image_path(self, entry: Dict[str, Any]) -> Path:
        return self.directory / entry["file"]

    def delete(self, entry: Dict[str, Any]) -> None:
        try:
            self.image_path(entry).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not delete image file: %s", exc)
        entries = [e for e in self._load_index() if e.get("file") != entry.get("file")]
        self._save_index(entries)

    def add_external(self, path: Path, meta: Optional[Dict[str, Any]] = None) -> None:
        """Register an image saved outside the store (e.g. re-analysis)."""
        entries = self._load_index()
        entries.append({
            "file": str(path),
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        })
        self._save_index(entries)

"""Shared utilities: paths, logging, time helpers."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

APP_NAME = "ISS SSTV Station"
APP_VERSION = "0.1.0"

# Project root: the directory that contains core/, sstv/, gui/.
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ISSSSTV_DATA_DIR", ROOT_DIR / "data"))
TLE_CACHE_DIR = DATA_DIR / "tle"
RECORDINGS_DIR = DATA_DIR / "recordings"
IMAGES_DIR = DATA_DIR / "images"
SETTINGS_PATH = DATA_DIR / "settings.json"


def ensure_dirs() -> None:
    for d in (DATA_DIR, TLE_CACHE_DIR, RECORDINGS_DIR, IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def setup_logging(level: int = logging.INFO, logfile: Path | None = None) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logging.getLogger("skyfield").setLevel(logging.WARNING)


def safe_filename(text: str) -> str:
    keep = (" ", "-", "_", ".")
    return "".join(c if c.isalnum() or c in keep else "_" for c in text).strip()

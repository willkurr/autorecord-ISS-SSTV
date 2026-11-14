"""Keplerian element acquisition and caching.

Elements are fetched from AMSAT's daily TLE feed
(https://www.amsat.org/tle/dailytle.txt) and cached on disk so the
application still works offline.  The ISS is identified by its NORAD
catalog number (25544) rather than by name.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from util import TLE_CACHE_DIR

log = logging.getLogger(__name__)

AMSAT_DAILY_TLE_URL = "https://www.amsat.org/tle/dailytle.txt"
ISS_CATALOG_NUMBER = 25544  # NORAD catalog ID of the International Space Station

_LINE1_RE = re.compile(r"^1 \d{5}[UAC] ")
_LINE2_RE = re.compile(r"^2 \d{5} ")


@dataclass(frozen=True)
class TLE:
    name: str
    line1: str
    line2: str

    @property
    def catalog_number(self) -> int:
        return int(self.line1[2:7])


def parse_tles(text: str) -> Dict[int, TLE]:
    """Parse a three-line-element (TLE) file into {catalog_number: TLE}."""
    sats: Dict[int, TLE] = {}
    lines = [ln.rstrip() for ln in text.splitlines()]
    i = 0
    while i < len(lines) - 1:
        if _LINE1_RE.match(lines[i]) and _LINE2_RE.match(lines[i + 1]):
            name = lines[i - 1].strip() if i >= 1 else "UNKNOWN"
            tle = TLE(name=name, line1=lines[i], line2=lines[i + 1])
            sats[tle.catalog_number] = tle
            i += 2
        else:
            i += 1
    return sats


class TLEProvider:
    """Downloads, caches and hands out TLE sets, with offline fallback."""

    def __init__(self, cache_dir: Path = TLE_CACHE_DIR,
                 url: str = AMSAT_DAILY_TLE_URL,
                 max_age_hours: float = 6.0,
                 timeout: float = 15.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "dailytle.txt"
        self.url = url
        self.max_age_hours = max_age_hours
        self.timeout = timeout
        self._sats: Dict[int, TLE] = {}
        self._last_fetch: float = 0.0

    # ------------------------------------------------------------------ cache
    def _cache_age_hours(self) -> Optional[float]:
        if not self.cache_file.exists():
            return None
        age_s = time.time() - self.cache_file.stat().st_mtime
        return age_s / 3600.0

    def cache_age_text(self) -> str:
        age = self._cache_age_hours()
        if age is None:
            return "no cache"
        if age < 1.0:
            return f"cached {int(age * 60)} min ago"
        return f"cached {age:.1f} h ago"

    # ------------------------------------------------------------------ fetch
    def update(self, force: bool = False) -> Dict[int, TLE]:
        """Fetch fresh elements when the cache is stale; fall back to cache."""
        age = self._cache_age_hours()
        fresh = age is not None and age < self.max_age_hours
        if fresh and not force and self._sats:
            return self._sats
        try:
            resp = requests.get(self.url, timeout=self.timeout)
            resp.raise_for_status()
            sats = parse_tles(resp.text)
            if ISS_CATALOG_NUMBER not in sats:
                raise ValueError("ISS (25544) missing from downloaded TLE set")
            self.cache_file.write_text(resp.text, encoding="utf-8")
            self._sats = sats
            self._last_fetch = time.time()
            log.info("Fetched %d TLEs from %s", len(sats), self.url)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully offline
            log.warning("TLE download failed (%s); trying cache", exc)
            if not self.cache_file.exists():
                raise RuntimeError(
                    "Could not download TLEs and no cache exists. "
                    "Check network connectivity."
                ) from exc
            self._sats = parse_tls_cache(self.cache_file)
        return self._sats

    # ------------------------------------------------------------------ access
    def get(self, catalog_number: int) -> TLE:
        if not self._sats:
            self.update()
        if catalog_number not in self._sats:
            raise KeyError(f"Catalog number {catalog_number} not in TLE set")
        return self._sats[catalog_number]

    def get_iss(self) -> TLE:
        return self.get(ISS_CATALOG_NUMBER)


def parse_tls_cache(path: Path) -> Dict[int, TLE]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sats = parse_tles(text)
    if not sats:
        raise RuntimeError(f"TLE cache {path} is empty or unparseable")
    return sats

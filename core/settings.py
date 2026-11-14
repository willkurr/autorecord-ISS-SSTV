"""Application settings, persisted as JSON in the data directory."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

from util import SETTINGS_PATH

log = logging.getLogger(__name__)


@dataclass
class Settings:
    # ground station
    lat_deg: float = 40.0
    lon_deg: float = -105.0
    alt_m: float = 0.0
    min_el_deg: float = 10.0
    predict_hours: float = 96.0
    # radio
    nominal_freq_hz: float = 145_800_000.0
    tuning_offset_hz: float = 20_000.0
    fine_offset_hz: float = 0.0
    ppm: int = 0
    gain_auto: bool = True
    manual_gain_db: float = 32.8
    channel_bw_hz: float = 15_000.0
    deemphasis: bool = True
    doppler_correction: bool = True
    monitor_audio: bool = False
    # pass handling
    auto_record: bool = True
    preroll_s: float = 15.0
    postroll_s: float = 15.0
    # TLE
    tle_url: str = "https://www.amsat.org/tle/dailytle.txt"
    tle_max_age_hours: float = 6.0
    # source: 'rtlsdr' | 'soundcard' | 'sim'
    source_type: str = "rtlsdr"
    # simulator
    sim_snr_db: float = 20.0
    sim_speed: float = 1.0

    # ------------------------------------------------------------------
    def validate(self) -> None:
        assert -90.0 <= self.lat_deg <= 90.0, "latitude out of range"
        assert -180.0 <= self.lon_deg <= 180.0, "longitude out of range"
        assert 0.0 <= self.min_el_deg < 90.0, "min elevation out of range"
        assert 1e6 < self.nominal_freq_hz < 30e9, "frequency out of range"
        assert 0.0 <= self.tuning_offset_hz < 100e3, "tuning offset out of range"
        assert self.source_type in ("rtlsdr", "soundcard", "sim")


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    s = Settings()
    if path.exists():
        try:
            data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            names = {f.name for f in fields(Settings)}
            for k, v in data.items():
                if k in names:
                    setattr(s, k, v)
            log.info("Settings loaded from %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load settings (%s); using defaults", exc)
    else:
        log.info("No settings file; using defaults")
    s.validate()
    return s


def save_settings(s: Settings, path: Path = SETTINGS_PATH) -> None:
    s.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
    log.info("Settings saved to %s", path)

"""Pass recorder: incremental WAV writing with a JSON metadata sidecar."""
from __future__ import annotations

import json
import logging
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from util import APP_VERSION, safe_filename

log = logging.getLogger(__name__)


class PassRecorder:
    """Writes the demodulated pass audio to disk as it streams in."""

    def __init__(self, directory: Path, satellite: str, freq_hz: float,
                 fs: float = 48_000.0, meta: Optional[Dict[str, Any]] = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H_%M_%SZ")
        base = safe_filename(f"{stamp}_{satellite}_{freq_hz / 1e3:.0f}kHz")
        self.wav_path = self.directory / f"{base}.wav"
        self.meta_path = self.directory / f"{base}.json"
        self.fs = float(fs)
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.update({
            "app_version": APP_VERSION,
            "satellite": satellite,
            "frequency_hz": freq_hz,
            "sample_rate": int(self.fs),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        })
        self._wav: Optional[wave.Wave_write] = None
        self._frames = 0
        self._images: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def start(self) -> Path:
        self._wav = wave.open(str(self.wav_path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(int(self.fs))
        log.info("Recording pass audio to %s", self.wav_path)
        return self.wav_path

    def write(self, audio: np.ndarray) -> None:
        if self._wav is None:
            return
        audio = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
        pcm = (audio * 32767.0).astype("<i2").tobytes()
        self._wav.writeframes(pcm)
        self._frames += len(audio)

    def note_image(self, filename: str, info: Dict[str, Any]) -> None:
        self._images.append({"file": filename, **info})

    # ------------------------------------------------------------------
    def close(self) -> Optional[Path]:
        if self._wav is None:
            return None
        self.meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
        self.meta["frames"] = self._frames
        self.meta["duration_s"] = self._frames / self.fs
        self.meta["images"] = self._images
        self._wav.close()
        self._wav = None
        self.meta_path.write_text(
            json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        log.info("Recording closed: %s (%.1f s, %d image(s))",
                 self.wav_path.name, self.meta["duration_s"], len(self._images))
        return self.wav_path

    # context manager convenience
    def __enter__(self) -> "PassRecorder":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

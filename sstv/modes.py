"""SSTV mode registry, keyed by VIS code.

Decoders for new modes register themselves here (see ``sstv.robot36``),
so the detector stays generic: it reports a VIS code and the registry
decides whether we can decode the transmission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class SSTVMode:
    name: str
    vis_code: int
    width: int
    height: int
    duration_s: float
    decoder_factory: Callable  # (fs, start_index) -> line decoder


_MODES: Dict[int, SSTVMode] = {}


def register_mode(mode: SSTVMode) -> None:
    _MODES[mode.vis_code] = mode


def mode_for_vis(vis_code: int) -> Optional[SSTVMode]:
    return _MODES.get(vis_code)


def all_modes() -> List[SSTVMode]:
    return sorted(_MODES.values(), key=lambda m: m.vis_code)


# Common SSTV tone frequencies (Hz).
FREQ_SYNC = 1200.0       # sync pulse / VIS start+stop / leader break
FREQ_BLACK = 1500.0      # black level
FREQ_WHITE = 2300.0      # white level
FREQ_LEADER = 1900.0     # calibration leader tone
FREQ_VIS_BIT1 = 1100.0
FREQ_VIS_BIT0 = 1300.0


def freq_to_level(freq: float) -> float:
    """Map tone frequency (Hz) to 0..255 video level."""
    return (freq - FREQ_BLACK) / (FREQ_WHITE - FREQ_BLACK) * 255.0


def level_to_freq(level: float) -> float:
    """Map 0..255 video level to tone frequency (Hz)."""
    return FREQ_BLACK + (FREQ_WHITE - FREQ_BLACK) * level / 255.0

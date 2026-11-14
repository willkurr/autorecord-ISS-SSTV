"""SSTV decoder events, shared by line decoders, the engine and the GUI."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sstv.modes import SSTVMode


class EventType(enum.Enum):
    IMAGE_START = 1     # an image header was detected; decoding begins
    LINE = 2            # one (or a re-finalized) image row is available
    IMAGE_COMPLETE = 3  # the image is finished (or hopelessly lost sync)


@dataclass
class SSTVEvent:
    type: EventType
    mode: Optional[SSTVMode] = None
    line: Optional[int] = None            # row index for LINE events
    row: Optional[np.ndarray] = None      # uint8 (width, 3) RGB row
    image: Optional[np.ndarray] = None    # uint8 (height, width, 3) for COMPLETE
    truncated: bool = False
    sync_quality: float = 1.0             # syncs found / syncs expected
    start_index: int = 0                  # global audio sample where image began

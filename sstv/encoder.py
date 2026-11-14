"""Robot 36 SSTV encoder.

Used for round-trip tests and for the pass simulator; timings follow the
published Robot 36 specification (identical to the pySSTV reference
implementation, which interoperates with MMSSTV/QSSTV).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from sstv.modes import (FREQ_BLACK, FREQ_LEADER, FREQ_SYNC, FREQ_VIS_BIT0,
                        FREQ_VIS_BIT1, FREQ_WHITE, level_to_freq)

# Robot 36 timing (ms)
SYNC_MS = 9.0
PORCH_MS = 3.0
Y_SCAN_MS = 88.0
SEP_MS = 4.5
CHROMA_PORCH_MS = 1.5
C_SCAN_MS = 44.0
LINE_MS = SYNC_MS + PORCH_MS + Y_SCAN_MS + SEP_MS + CHROMA_PORCH_MS + C_SCAN_MS  # 150
WIDTH = 320
HEIGHT = 240
VIS_CODE = 0x08


def rgb_to_ycbcr(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JPEG/PIL-style full-range YCbCr conversion (Y,Cb,Cr all 0..255)."""
    r = image[..., 0].astype(np.float64)
    g = image[..., 1].astype(np.float64)
    b = image[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _vis_segments(vis_code: int) -> List[Tuple[float, float]]:
    """(freq, ms) pairs for the calibration header, per the SSTV spec."""
    segs: List[Tuple[float, float]] = [
        (FREQ_LEADER, 300.0),   # first leader tone
        (FREQ_SYNC, 10.0),      # break
        (FREQ_LEADER, 300.0),   # second leader tone
        (FREQ_SYNC, 30.0),      # VIS start bit
    ]
    bits = [(vis_code >> k) & 1 for k in range(7)]  # LSB first
    for bit in bits:
        segs.append((FREQ_VIS_BIT1 if bit else FREQ_VIS_BIT0, 30.0))
    ones = sum(bits)
    parity = 1 if ones % 2 == 1 else 0  # even parity overall
    segs.append((FREQ_VIS_BIT1 if parity else FREQ_VIS_BIT0, 30.0))
    segs.append((FREQ_SYNC, 30.0))  # VIS stop bit
    return segs


def _robot36_segments(image: np.ndarray) -> List[Tuple[float, float]]:
    """(freq, ms) pairs for the image body of a Robot 36 transmission."""
    if image.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"expected ({HEIGHT}, {WIDTH}, 3) image, got {image.shape}")
    y, cb, cr = rgb_to_ycbcr(image)
    y_px = Y_SCAN_MS / WIDTH
    c_px = C_SCAN_MS / WIDTH
    segs: List[Tuple[float, float]] = []
    for line in range(HEIGHT):
        chroma_is_cr = (line % 2) == 0  # even lines carry Cr
        segs.append((FREQ_SYNC, SYNC_MS))
        segs.append((FREQ_BLACK, PORCH_MS))
        for col in range(WIDTH):
            segs.append((level_to_freq(y[line, col]), y_px))
        if chroma_is_cr:
            segs.append((FREQ_BLACK, SEP_MS))   # separator: 1500 Hz -> Cr next
        else:
            segs.append((FREQ_WHITE, SEP_MS))   # separator: 2300 Hz -> Cb next
        segs.append((FREQ_LEADER, CHROMA_PORCH_MS))
        chan = cr if chroma_is_cr else cb
        for col in range(WIDTH):
            segs.append((level_to_freq(chan[line, col]), c_px))
    return segs


def _synthesize(segs: List[Tuple[float, float]], fs: float, amplitude: float) -> np.ndarray:
    """Continuous-phase sine synthesis of (freq, ms) segments."""
    total_ms = sum(ms for _, ms in segs)
    n_total = int(round(total_ms * fs / 1000.0))
    # segment sample boundaries (no cumulative rounding drift)
    bounds = np.cumsum([0.0] + [ms * fs / 1000.0 for _, ms in segs])
    freqs = np.empty(n_total)
    pos = 0
    for (f_hz, _), b0, b1 in zip(segs, bounds[:-1], bounds[1:]):
        i0, i1 = int(round(b0)), int(round(b1))
        i1 = max(i1, i0 + 1)  # every segment emits >= 1 sample
        i1 = min(i1, n_total)
        if i1 > i0:
            freqs[i0:i1] = f_hz
            pos = i1
    if pos < n_total:
        freqs[pos:] = segs[-1][0]
    phase = 2.0 * np.pi * np.cumsum(freqs) / fs
    return (amplitude * np.sin(phase)).astype(np.float32)


def encode_robot36(image: np.ndarray, fs: float = 48000.0,
                   amplitude: float = 0.85, include_header: bool = True) -> np.ndarray:
    """Encode a 320x240 RGB image as a Robot 36 SSTV audio signal."""
    segs: List[Tuple[float, float]] = []
    if include_header:
        segs += _vis_segments(VIS_CODE)
    segs += _robot36_segments(image)
    return _synthesize(segs, fs, amplitude)


def test_image(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A recognizable synthetic test image (color bars + gradients + text-ish blocks)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bars = [
        (255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
        (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0),
    ]
    n = len(bars)
    bar_w = width // n
    for i, rgb in enumerate(bars):
        img[:, i * bar_w:(i + 1) * bar_w] = rgb
    # vertical gradient overlay in the middle band
    yy = np.linspace(0, 255, width)
    band = slice(height // 3, 2 * height // 3)
    img[band, :, 0] = np.tile(yy, (height // 3, 1)).astype(np.uint8)
    # bright/dark corner patches
    img[0:40, 0:40] = 255
    img[-40:, -40:] = 0
    return img

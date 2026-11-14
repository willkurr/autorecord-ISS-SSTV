"""RTL-SDR source (pyrtlsdr).

The dongle is parked at a fixed center frequency (nominal - tuning offset)
and streams complex IQ; doppler is corrected downstream in software.
pyrtlsdr needs the librtlsdr native library and is therefore imported
lazily, so the rest of the application works without any SDR attached.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Tuple

import numpy as np

from core.sources.base import Chunk, SourceError, TimedSource

log = logging.getLogger(__name__)

# Shared, platform-specific setup instructions shown whenever pyrtlsdr or
# librtlsdr is missing (settings dialog "Check RTL-SDR", source errors).
RTLSDR_SETUP_INSTRUCTIONS = """\
The RTL-SDR driver stack is missing. Install it for your platform:

  Windows:
    1. Install the Zadig WinUSB driver for the dongle
       (https://zadig.akeo.ie - pick the RTL-SDR device, choose WinUSB).
    2. Place rtlsdr.dll next to the program or somewhere on PATH
       (included in the rtl-sdr Windows release build).
  macOS:
    brew install librtlsdr
  Linux (Debian/Ubuntu):
    sudo apt install librtlsdr-dev
  Linux (Fedora):
    sudo dnf install rtl-sdr-devel
  Linux (Arch):
    sudo pacman -S rtl-sdr

Then install the Python wrapper:
    pip install pyrtlsdr
"""

_NO_DEVICE_INSTRUCTIONS = """\
pyrtlsdr is installed, but no RTL-SDR dongle could be opened:

  * Check that the dongle is plugged in.
  * Windows: make sure the Zadig WinUSB driver is installed for it
    (https://zadig.akeo.ie).
  * Linux: another program may be using the dongle, or the DVB-TV
    kernel driver claimed it:
      echo "blacklist dvb_usb_rtl28xxu" | \\
        sudo tee /etc/modprobe.d/blacklist-rtl.conf
    then reboot (or: sudo rmmod dvb_usb_rtl28xxu).
"""

# smallest synchronous USB read the fallback ladder will try (samples)
MIN_READ_SAMPLES = 1024


def probe_rtlsdr() -> Tuple[bool, str]:
    """Check the RTL-SDR software stack and hardware, without blocking.

    Returns ``(ok, message)``: a human-readable success report, or a
    failure description that includes platform-specific setup
    instructions when the library/driver layer is missing.
    """
    try:
        from rtlsdr import RtlSdr  # noqa: PLC0415 - deliberate lazy import
    except Exception as exc:  # noqa: BLE001 - report any import failure
        log.debug("rtlsdr import failed: %s", exc)
        return False, ("Could not import pyrtlsdr (librtlsdr not found):\n"
                       f"  {exc}\n\n" + RTLSDR_SETUP_INSTRUCTIONS)
    try:
        sdr = RtlSdr()
    except Exception as exc:  # noqa: BLE001
        log.debug("RtlSdr() open failed: %s", exc)
        return False, (f"pyrtlsdr loaded, but no RTL-SDR dongle could be "
                       f"opened:\n  {exc}\n\n" + _NO_DEVICE_INSTRUCTIONS)
    try:
        # exercise the device with a short test read
        sdr.sample_rate = 240_000
        sdr.center_freq = 145_800_000.0
        samples = sdr.read_samples(8192)
        name = getattr(sdr, "device_name", "unknown device")
        sdr.close()
        n = 0 if samples is None else len(samples)
        if n == 0:
            return False, ("RTL-SDR opened but returned no samples.\n\n"
                           + _NO_DEVICE_INSTRUCTIONS)
        return True, (f"RTL-SDR is working.\n\nDevice: {name}\n"
                      f"Test read: {n} samples at 240 kS/s")
    except Exception as exc:  # noqa: BLE001
        try:
            sdr.close()
        except Exception:  # noqa: BLE001
            pass
        log.debug("RTL-SDR test read failed: %s", exc)
        return False, (f"RTL-SDR opened, but a test read failed:\n  {exc}\n\n"
                       + _NO_DEVICE_INSTRUCTIONS)


class RtlSdrSource(TimedSource):
    """Streams complex IQ from an RTL-SDR dongle."""

    def __init__(self,
                 center_freq_hz: float,
                 fs: float = 240_000.0,
                 chunk_samples: int = 24_000,
                 ppm: int = 0,
                 gain: object = "auto",
                 device_index: Optional[int] = None,
                 label: str = "rtlsdr"):
        super().__init__(fs, chunk_samples, is_iq=True, label=label)
        self.center_freq_hz = float(center_freq_hz)
        self.ppm = int(round(ppm))
        self.gain = gain
        self.device_index = device_index
        self._sdr = None

    # ------------------------------------------------------------------
    def _open(self):
        try:
            from rtlsdr import RtlSdr  # lazy: librtlsdr may be absent
        except Exception as exc:  # noqa: BLE001
            raise SourceError(
                "pyrtlsdr is not installed or librtlsdr was not found.\n"
                + RTLSDR_SETUP_INSTRUCTIONS
                + f"\n(import error: {exc})"
            ) from exc
        try:
            sdr = RtlSdr(device_index=self.device_index if self.device_index is not None else 0)
        except Exception as exc:  # noqa: BLE001
            raise SourceError(
                "Could not open RTL-SDR device.\n"
                + _NO_DEVICE_INSTRUCTIONS
                + f"\n(open error: {exc})"
            ) from exc
        # Sample rate and center frequency are essential: failure here is
        # fatal.  PPM correction and gain are optional on some
        # drivers/firmware (e.g. several librtlsdr builds reject ppm = 0),
        # so those failures only degrade gracefully with a warning.
        self._set_required(sdr, "sample_rate", self.fs)
        self._set_required(sdr, "center_freq", self.center_freq_hz)
        if self.ppm != 0:
            self._set_optional(sdr, "freq_correction", self.ppm, "PPM correction")
        if isinstance(self.gain, str):
            self._set_optional(sdr, "gain", self.gain, "gain")
        else:
            self._set_optional(sdr, "gain", float(self.gain), "gain")
        log.info("RTL-SDR opened: center=%.6f MHz, fs=%.0f Hz, ppm=%d, gain=%s",
                 self.center_freq_hz / 1e6, self.fs, self.ppm,
                 sdr.gain if not isinstance(self.gain, str) else "auto")
        self._sdr = sdr

    @staticmethod
    def _set_required(sdr, attr: str, value) -> None:
        try:
            setattr(sdr, attr, value)
        except Exception as exc:  # noqa: BLE001
            sdr.close()
            raise SourceError(
                f"Failed to set RTL-SDR {attr} to {value}: {exc}") from exc

    @staticmethod
    def _set_optional(sdr, attr: str, value, label: str) -> None:
        try:
            setattr(sdr, attr, value)
        except Exception as exc:  # noqa: BLE001 - non-fatal
            log.warning("Could not set RTL-SDR %s to %s (%s); continuing. "
                        "Reception still works, but frequency accuracy may "
                        "suffer - use the Fine offset setting to compensate.",
                        label, value, exc)

    # ------------------------------------------------------------------
    def _loop(self, sink: Callable[[Chunk], None]) -> None:
        self._open()
        assert self._sdr is not None
        # macOS's libusb reports LIBUSB_ERROR_OVERFLOW for large
        # synchronous reads even though the same size works on
        # Windows/Linux.  Start with the full chunk and shrink the USB
        # read size until it succeeds; full pipeline chunks are still
        # emitted regardless of the underlying read granularity.
        read_size = self.chunk_samples
        pending = np.zeros(0, dtype=np.complex64)
        while not self._stop_evt.is_set():
            try:
                samples = self._sdr.read_samples(read_size)
            except Exception as exc:  # noqa: BLE001
                if self._is_overflow(exc) and read_size > MIN_READ_SAMPLES:
                    read_size = max(MIN_READ_SAMPLES, read_size // 2)
                    log.warning("RTL-SDR USB overflow; retrying with "
                                "%d-sample reads", read_size)
                    time.sleep(0.02)  # let the device buffers drain
                    continue
                raise SourceError(f"RTL-SDR read failed: {exc}") from exc
            if samples is None or len(samples) == 0:
                time.sleep(0.01)
                continue
            pending = np.concatenate(
                [pending, np.asarray(samples, dtype=np.complex64).ravel()])
            while len(pending) >= self.chunk_samples:
                t = self._stamp()
                chunk = pending[:self.chunk_samples]
                pending = pending[self.chunk_samples:]
                sink(Chunk(data=chunk, t=t, is_iq=True, fs=self.fs))

    @staticmethod
    def _is_overflow(exc: Exception) -> bool:
        return "OVERFLOW" in str(exc).upper()

    def _cleanup(self) -> None:
        if self._sdr is not None:
            try:
                self._sdr.close()
            except Exception:  # noqa: BLE001
                pass
            self._sdr = None

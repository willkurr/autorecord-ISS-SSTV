"""RTL-SDR source (pyrtlsdr).

The dongle is parked at a fixed center frequency (nominal - tuning offset)
and streams complex IQ; doppler is corrected downstream in software.
pyrtlsdr needs the librtlsdr native library and is therefore imported
lazily, so the rest of the application works without any SDR attached.
"""
from __future__ import annotations

import logging
import queue
import threading
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
# how many overflow-driven close/reopen cycles before giving up
MAX_OVERFLOW_RECOVERIES = 10


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
        self.actual_fs: float = float(fs)
        self._requested_fs = float(fs)
        # reader thread -> bounded queue -> consumer thread: the reader
        # must never be blocked by DSP work, or the dongle's FIFO
        # overflows and samples are dropped
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._consumer_thread = None
        self._sink_cb: Optional[Callable[[Chunk], None]] = None
        self.dropped_samples = 0

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
        self._set_required(sdr, "sample_rate", self._requested_fs)
        # The rate the hardware actually locked onto can differ from the one
        # requested (e.g. 226 kS/s on some dongles/ports).  Decoding every
        # downstream stage assumes the requested rate, so a mismatch sounds
        # like crackling/pitch-shifted audio and destroys SSTV timing.
        actual = None
        try:
            actual = float(sdr.sample_rate)
        except Exception:  # noqa: BLE001
            pass
        if actual is not None and abs(actual - self._requested_fs) > 1.0:
            self.actual_fs = actual
            self.fs = actual  # timestamps and chunk metadata follow reality
            correction = abs(actual - self._requested_fs) / self._requested_fs
            if correction > 0.02:
                log.warning(
                    "RTL-SDR running at %.0f Hz instead of the requested "
                    "%.0f Hz (%.1f%% off). Audio will be pitch-shifted and "
                    "SSTV sync may fail; the DSP chain now compensates for "
                    "the real rate.", actual, self._requested_fs,
                    correction * 100.0)
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

    def chunk_fs(self) -> float:
        """Sample rate the stream is actually produced at.

        Hardware may not honour the requested rate exactly; the raw chunk
        timestamps and any downstream rate maths must use what the device
        really delivers.
        """
        return float(getattr(self, "actual_fs", self.fs))

    # ------------------------------------------------------------------
    def _reader_loop(self) -> None:
        """Dedicated USB drain thread.

        This loop must do *nothing* but read and enqueue: the dongle
        streams continuously into a FIFO only ~15 ms deep, so any pause
        longer than that (e.g. running the DSP chain inline) silently
        drops samples - the cause of crackling audio, glitchy recordings
        and broken SSTV sync.  Keeping this loop tight keeps the FIFO
        drained.
        """
        self._open()
        assert self._sdr is not None
        # Some platforms (macOS, and Linux with certain controllers) fail
        # synchronous reads of ~24 KB with LIBUSB_ERROR_OVERFLOW.  Shrink
        # the USB read size until reads succeed; an overflow also poisons
        # the handle, so each one triggers a close/reopen before the next
        # read (continuing on an overflowed handle can crash natively).
        read_size = self.chunk_samples
        overflows = 0
        while not self._stop_evt.is_set():
            try:
                samples = self._sdr.read_samples(read_size)
            except Exception as exc:  # noqa: BLE001
                if not self._is_overflow(exc):
                    raise SourceError(f"RTL-SDR read failed: {exc}") from exc
                overflows += 1
                if overflows > MAX_OVERFLOW_RECOVERIES:
                    raise SourceError(
                        "RTL-SDR kept overflowing even after reopening the "
                        f"device at {read_size}-sample reads ({overflows} "
                        "attempts). This usually means a USB bandwidth or "
                        "power problem: try a different USB port (directly "
                        "on the computer, not a hub or adapter) and a "
                        "shorter cable."
                    ) from exc
                if read_size > MIN_READ_SAMPLES:
                    read_size = max(MIN_READ_SAMPLES, read_size // 2)
                log.warning(
                    "RTL-SDR USB overflow; reopening the device with "
                    "%d-sample reads (attempt %d of %d)",
                    read_size, overflows, MAX_OVERFLOW_RECOVERIES)
                time.sleep(0.1)  # let the device settle before reopening
                try:
                    self._reopen()  # same config, fresh handle, size kept
                except Exception as reopen_exc:  # noqa: BLE001
                    raise SourceError(
                        f"Could not reopen the RTL-SDR after an overflow: "
                        f"{reopen_exc}") from reopen_exc
                continue
            if samples is None or len(samples) == 0:
                continue
            overflows = 0
            self._enqueue(np.asarray(samples, dtype=np.complex64).ravel())

    def _enqueue(self, samples: np.ndarray) -> None:
        """Queue raw samples for the consumer; drop oldest if it overflows."""
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
                self.dropped_samples += len(dropped)
                log.warning("RTL-SDR receiver queue overflow: dropped "
                            "%d samples (the computer cannot keep up with "
                            "the DSP chain)", len(dropped))
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(samples)
        except queue.Full:
            self.dropped_samples += len(samples)

    def _consume_loop(self) -> None:
        """Assemble queued raw reads into full pipeline chunks and hand
        them to the sink."""
        buffer = np.zeros(self.chunk_samples * 4, dtype=np.complex64)
        filled = 0
        while not self._stop_evt.is_set() or not self._queue.empty():
            try:
                samples = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if filled + samples.size > buffer.size:
                grown = np.empty(max(buffer.size,
                                     (filled + samples.size) * 2),
                                 dtype=np.complex64)
                grown[:filled] = buffer[:filled]
                buffer = grown
            buffer[filled:filled + samples.size] = samples
            filled += samples.size
            while filled >= self.chunk_samples:
                t = self._stamp()
                sink_chunk = Chunk(data=buffer[:self.chunk_samples].copy(),
                                   t=t, is_iq=True, fs=self.actual_fs)
                buffer[:filled - self.chunk_samples] = \
                    buffer[self.chunk_samples:filled]
                filled -= self.chunk_samples
                try:
                    self._sink_cb(sink_chunk)
                except Exception:  # noqa: BLE001
                    log.exception("sink failed")

    # ------------------------------------------------------------------
    def start(self, sink: Callable[[Chunk], None]) -> None:
        if self._thread is not None or self._consumer_thread is not None:
            raise SourceError("source already running")
        self._sink_cb = sink
        self._stop_evt.clear()
        self._error = None
        self._consumer_thread = threading.Thread(
            target=self._consume_loop, name=f"cons-{self.label}", daemon=True)
        self._consumer_thread.start()
        self._thread = threading.Thread(
            target=self._reader_run, name=f"src-{self.label}", daemon=True)
        self._thread.start()

    def _reader_run(self) -> None:
        try:
            self._reader_loop()
        except Exception as exc:  # noqa: BLE001 - surface to stop()
            self._error = exc
            log.exception("Source %s failed", self.label)
        finally:
            self._stop_evt.set()  # unblock (and end) the consumer
            try:
                self._cleanup()
            finally:
                if self.on_end is not None:
                    try:
                        self.on_end()
                    except Exception:  # noqa: BLE001
                        log.exception("source end callback failed")

    def _stamp(self) -> float:
        """Timestamps follow the rate the hardware actually delivers."""
        if self._t0 is None:
            self._t0 = time.time()
        t = self._t0 + self._samples / self.actual_fs
        self._samples += self.chunk_samples
        return t

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_evt.set()
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=timeout)
            self._consumer_thread = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._error is not None:
            raise self._error

    @property
    def running(self) -> bool:
        return ((self._thread is not None and self._thread.is_alive())
                or (self._consumer_thread is not None
                    and self._consumer_thread.is_alive()))

    @staticmethod
    def _is_overflow(exc: Exception) -> bool:
        return "OVERFLOW" in str(exc).upper()

    def _reopen(self) -> None:
        """Close and reopen the device, preserving configuration.

        A fresh handle is essential after a USB overflow: continuing to
        read from a handle whose buffer overflowed is documented to
        misbehave on some platforms (including native crashes on macOS).
        """
        self._cleanup()
        self._open()

    def _cleanup(self) -> None:
        if self._sdr is not None:
            try:
                self._sdr.close()
            except Exception:  # noqa: BLE001
                pass
            self._sdr = None

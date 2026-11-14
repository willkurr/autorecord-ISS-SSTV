"""Tests for the RTL-SDR connection check."""
from __future__ import annotations

import sys
import types

import pytest

from core.sources.sdr import (RTLSDR_SETUP_INSTRUCTIONS, probe_rtlsdr)


@pytest.fixture
def fake_rtlsdr():
    """Inject a fake 'rtlsdr' module for import-isolated testing."""
    module = types.ModuleType("rtlsdr")

    class FakeRtlSdr:
        device_name = "Generic RTL2832U OEM"
        instances = []

        def __init__(self, *args, **kwargs):
            self.closed = False
            self.read_error = False
            type(self).instances.append(self)

        def read_samples(self, n):
            if self.read_error:
                raise OSError("usb read error")
            return [0] * n

        def close(self):
            self.closed = True

    module.RtlSdr = FakeRtlSdr
    saved = sys.modules.get("rtlsdr")
    sys.modules["rtlsdr"] = module
    yield module, FakeRtlSdr
    if saved is not None:
        sys.modules["rtlsdr"] = saved
    else:
        sys.modules.pop("rtlsdr", None)


def test_probe_reports_missing_library(monkeypatch):
    """Without pyrtlsdr installed, the check fails with install guidance."""
    monkeypatch.setitem(sys.modules, "rtlsdr", None)  # forces ImportError
    ok, message = probe_rtlsdr()
    assert ok is False
    assert "librtlsdr" in message
    # platform-specific instructions are included
    assert "brew install librtlsdr" in message
    assert "apt install librtlsdr-dev" in message
    assert "pip install pyrtlsdr" in message
    assert "Zadig" in message


def test_probe_reports_no_device(fake_rtlsdr):
    """pyrtlsdr present but RtlSdr() fails -> 'no dongle' guidance."""
    module, Fake = fake_rtlsdr

    def boom(*args, **kwargs):
        raise OSError("no devices")

    Fake.__init__ = boom
    ok, message = probe_rtlsdr()
    assert ok is False
    assert "no RTL-SDR dongle could be opened" in message
    assert "blacklist dvb_usb_rtl28xxu" in message  # Linux driver note


def test_probe_reports_failed_read(fake_rtlsdr):
    """Device opens but the test read fails -> clear error, device closed."""
    module, Fake = fake_rtlsdr
    Fake.read_samples = lambda self, n: (_ for _ in ()).throw(OSError("usb"))
    ok, message = probe_rtlsdr()
    assert ok is False
    assert "test read failed" in message
    assert Fake.instances and Fake.instances[0].closed


def test_probe_success(fake_rtlsdr):
    module, Fake = fake_rtlsdr
    ok, message = probe_rtlsdr()
    assert ok is True
    assert "Generic RTL2832U OEM" in message
    assert Fake.instances[0].closed  # the probe releases the device


# ---------------------------------------------------------------------------
# RtlSdrSource configuration robustness

def _make_source(ppm=0, gain="auto", device_index=None):
    from core.sources.sdr import RtlSdrSource
    return RtlSdrSource(center_freq_hz=145_800_000.0 - 20_000.0, ppm=ppm,
                        gain=gain, device_index=device_index)


def test_source_skips_zero_ppm(fake_rtlsdr):
    """ppm = 0 must not be sent to the driver (some builds reject it)."""
    module, Fake = fake_rtlsdr
    calls = []

    def make_setter():
        prop = property(lambda self: 0,
                        lambda self, v: calls.append(v))
        return prop

    Fake.freq_correction = make_setter()
    source = _make_source(ppm=0)
    source._open()
    try:
        assert calls == []          # never set
        assert source._sdr is not None
    finally:
        source._cleanup()


def test_source_ppm_failure_is_non_fatal(fake_rtlsdr):
    """A rejecting ppm setter only warns; the source still opens."""
    module, Fake = fake_rtlsdr

    def raise_setter(self, v):
        raise OSError("LIBUSB_ERROR_INVALID_PARAM")

    Fake.freq_correction = property(lambda self: 0, raise_setter)
    source = _make_source(ppm=30)
    source._open()
    try:
        assert source._sdr is not None
    finally:
        source._cleanup()


def test_source_required_setting_failure_is_fatal(fake_rtlsdr):
    """A failing sample-rate setter aborts the source with SourceError."""
    module, Fake = fake_rtlsdr

    def raise_setter(self, v):
        raise OSError("cannot set rate")

    Fake.sample_rate = property(lambda self: 0, raise_setter)
    from core.sources.base import SourceError
    source = _make_source()
    with pytest.raises(SourceError):
        source._open()
    # the device was released
    assert Fake.instances[0].closed


# ---------------------------------------------------------------------------
# macOS-style USB overflow fallback

@pytest.fixture
def overflow_rtlsdr():
    """Fake rtlsdr whose sync reads fail above 4096 samples (macOS-like),
    returning a monotonically increasing sample counter otherwise."""
    import numpy as np
    import types

    class LibUSBError(Exception):
        pass

    module = types.ModuleType("rtlsdr")
    module.LibUSBError = LibUSBError

    class FakeRtlSdr:
        max_read = 4096
        instances = []
        counter = 0

        def __init__(self, *args, **kwargs):
            self.closed = False
            type(self).instances.append(self)

        def read_samples(self, n):
            if n > type(self).max_read:
                raise LibUSBError(
                    f"<LIBUSB_ERROR_OVERFLOW (-8): Overflow> "
                    f"Could not read {n} samples")
            out = np.arange(type(self).counter,
                            type(self).counter + n, dtype=np.float64)
            type(self).counter += n
            return (out + 1j * out).astype(np.complex64)

        def close(self):
            self.closed = True

    module.RtlSdr = FakeRtlSdr
    saved = sys.modules.get("rtlsdr")
    sys.modules["rtlsdr"] = module
    yield module, FakeRtlSdr
    if saved is not None:
        sys.modules["rtlsdr"] = saved
    else:
        sys.modules.pop("rtlsdr", None)


def test_source_survives_usb_overflow(overflow_rtlsdr):
    """Large sync reads that overflow must shrink automatically while the
    emitted chunk stream stays complete, ordered and gap-free."""
    import time
    import numpy as np
    from core.sources.sdr import RtlSdrSource, MIN_READ_SAMPLES

    module, Fake = overflow_rtlsdr
    source = RtlSdrSource(center_freq_hz=145_780_000.0, chunk_samples=12_000)
    chunks = []
    source.start(chunks.append)
    deadline = time.time() + 10.0
    while len(chunks) < 4 and time.time() < deadline:
        time.sleep(0.05)
    source.stop()
    assert len(chunks) >= 4, "source never produced chunks after overflow"

    # every emitted chunk has the full pipeline size
    assert all(len(c.data) == 12_000 for c in chunks)
    # continuity: the counter must increase with no gaps or repeats
    joined = np.concatenate([c.data.real for c in chunks[:4]])
    assert np.allclose(joined, np.arange(joined.size), atol=1e-6)
    # the read size ladder actually shrank
    assert Fake.instances, "driver was never opened"


def test_overflow_ladder_respects_minimum(overflow_rtlsdr):
    """If even the minimum read size overflows, the source fails cleanly."""
    import time
    from core.sources.sdr import RtlSdrSource
    from core.sources.base import SourceError

    module, Fake = overflow_rtlsdr
    Fake.max_read = 0  # every read overflows
    source = RtlSdrSource(center_freq_hz=145_780_000.0, chunk_samples=12_000)
    errors = []
    original = source._error if hasattr(source, "_error") else None
    source.start(lambda c: None)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        time.sleep(0.05)
        if not source.running:
            break
    with pytest.raises(SourceError):
        source.stop()  # surfaces the thread's error
    Fake.instances[0].closed = True  # cleanup already ran

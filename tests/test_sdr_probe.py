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

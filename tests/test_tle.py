from __future__ import annotations

from core.tle import ISS_CATALOG_NUMBER, TLEProvider, parse_tles
from tests.conftest import AMSAT_SNIPPET, ISS_TLE


def test_parse_finds_iss_and_others():
    sats = parse_tles(AMSAT_SNIPPET)
    assert set(sats) == {7530, 25544, 27607}
    iss = sats[ISS_CATALOG_NUMBER]
    assert iss.name == "ISS"
    assert iss.line1 == ISS_TLE.line1
    assert iss.line2 == ISS_TLE.line2


def test_parse_ignores_garbage():
    text = AMSAT_SNIPPET + "\nrandom text\nmore junk\n1 not a tle\n"
    sats = parse_tles(text)
    assert ISS_CATALOG_NUMBER in sats


def test_provider_cache_roundtrip(tmp_path):
    prov = TLEProvider(cache_dir=tmp_path, max_age_hours=6.0)
    prov.cache_file.write_text(AMSAT_SNIPPET, encoding="utf-8")
    # Force the offline path by pointing at an unreachable URL.
    prov.url = "http://127.0.0.1:1/never"
    sats = prov.update(force=True)
    assert sats[ISS_CATALOG_NUMBER].name == "ISS"
    assert prov.get_iss().catalog_number == ISS_CATALOG_NUMBER


def test_provider_requires_network_or_cache(tmp_path):
    prov = TLEProvider(cache_dir=tmp_path, max_age_hours=0.0)
    prov.url = "http://127.0.0.1:1/never"
    try:
        prov.update(force=True)
        assert False, "expected failure with no cache and no network"
    except RuntimeError:
        pass

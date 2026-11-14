#!/usr/bin/env python3
"""ISS SSTV Station - ISS slow-scan TV tracking, recording and decoding.

Usage:
    python main.py [gui] [--simulate]     launch the GUI (default)
    python main.py predict [--lat LAT --lon LON --hours H --min-el DEG]
    python main.py decode FILE.wav [FILE.wav ...]
    python main.py demo [--snr DB]       headless simulated pass decode
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from util import APP_NAME, APP_VERSION, ensure_dirs, setup_logging


def cmd_gui(args) -> int:
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    window = MainWindow(simulate=args.simulate)
    window.show()
    return app.exec()


def cmd_predict(args) -> int:
    from core.tle import ISS_CATALOG_NUMBER, TLEProvider
    from core.tracker import Tracker
    ensure_dirs()
    prov = TLEProvider()
    tle = prov.get(ISS_CATALOG_NUMBER)
    tr = Tracker(tle)
    tr.set_station(args.lat, args.lon, args.alt)
    now = datetime.now(timezone.utc)
    passes = tr.next_passes(now, hours=args.hours, min_el_deg=args.min_el)
    local = now.astimezone().tzinfo
    print(f"ISS passes above {args.min_el:.0f} deg from "
          f"{args.lat:.4f}, {args.lon:.4f} (next {args.hours:.0f} h) - "
          f"times in {local}")
    print(f"{'AOS (local)':<22}{'dur':>7}{'max el':>8}{'AOS az':>8}{'LOS az':>8}")
    for p in passes:
        print(f"{p.aos.astimezone(local).strftime('%a %Y-%m-%d %H:%M:%S'):<22}"
              f"{int(p.duration_s // 60):>4}m{int(p.duration_s % 60):02d}"
              f"{p.max_el_deg:>7.1f}f"
              f"{p.aos_az_deg:>7.0f}f"
              f"{p.los_az_deg:>7.0f}f".replace("f", "°"))
    return 0


def cmd_decode(args) -> int:
    from core.images import ImageStore
    from sstv.engine import decode_file
    ensure_dirs()
    store = ImageStore()
    total = 0
    for wav in args.files:
        path = Path(wav)
        print(f"Decoding {path} ...")
        completed = decode_file(path)
        for ev in completed:
            out = store.save(ev.image, meta={
                "mode": ev.mode.name if ev.mode else None,
                "truncated": ev.truncated,
                "sync_quality": ev.sync_quality,
                "source": "cli",
                "recording": path.name,
            })
            print(f"  -> {out} ({'complete' if not ev.truncated else 'partial'})")
            total += 1
        if not completed:
            print("  -> no images found")
    print(f"{total} image(s) saved")
    return 0 if total else 1


def cmd_demo(args) -> int:
    """Headless end-to-end demo: simulate a pass and decode it."""
    from core.doppler import DopplerModel
    from core.dsp import ReceiveChain
    from core.sources.sim import SimulatorSource
    from core.tle import ISS_CATALOG_NUMBER, TLEProvider
    from core.tracker import PassInfo, Tracker
    from sstv.encoder import test_image
    from sstv.engine import SSTVEngine
    from sstv.events import EventType

    ensure_dirs()
    print("Fetching ISS elements...")
    prov = TLEProvider()
    tle = prov.get(ISS_CATALOG_NUMBER)
    tr = Tracker(tle)
    tr.set_station(40.0, -105.0, 1600.0)
    now = datetime.now(timezone.utc)
    passes = tr.next_passes(now, hours=24.0, min_el_deg=20.0)
    if not passes:
        print("No suitable pass found in the next 24 h")
        return 1
    p = passes[0]
    # simulate just a 2-minute window around culmination to keep it quick
    tca = p.aos + timedelta(seconds=p.duration_s / 2.0)
    pinfo = PassInfo(aos=tca - timedelta(seconds=60),
                     los=tca + timedelta(seconds=60),
                     max_el_deg=p.max_el_deg, aos_az_deg=p.aos_az_deg,
                     los_az_deg=p.los_az_deg)
    print(f"Simulating pass culminating {tca.strftime('%H:%M:%SZ')} "
          f"at {p.max_el_deg:.0f} deg, SNR {args.snr:.0f} dB")
    sim = SimulatorSource(tr, pinfo, snr_db=args.snr, speed=20.0,
                          preroll_s=5.0, postroll_s=5.0,
                          first_image_delay_s=20.0)
    dop = DopplerModel(tr, sim.freq_hz)
    chain = ReceiveChain(
        doppler_segment=lambda t0, t1, n: dop.segment(
            sim.sim_time(t0), sim.sim_time(t1), n))
    events = []
    engine = SSTVEngine(48_000.0, on_event=events.append)

    def sink(chunk):
        engine.process(chain.process(chunk.data, chunk.t))

    sim.start(sink)
    import time as _t
    while sim.running:
        _t.sleep(0.2)
    sim.stop()
    engine.flush()

    comps = [e for e in events if e.type == EventType.IMAGE_COMPLETE]
    print(f"Decoded {len(comps)} image(s)")
    if comps:
        from core.images import ImageStore
        store = ImageStore()
        for ev in comps:
            out = store.save(ev.image, meta={
                "mode": "Robot 36", "truncated": ev.truncated,
                "sync_quality": ev.sync_quality, "source": "demo"})
            print(f"  saved {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="iss-sstv", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version",
                        version=f"{APP_NAME} {APP_VERSION}")
    sub = parser.add_subparsers(dest="cmd")

    p_gui = sub.add_parser("gui", help="launch the GUI (default)")
    p_gui.add_argument("--simulate", action="store_true",
                       help="use the pass simulator instead of hardware")
    p_gui.set_defaults(func=cmd_gui)

    p_pred = sub.add_parser("predict", help="print pass predictions")
    p_pred.add_argument("--lat", type=float, default=40.0)
    p_pred.add_argument("--lon", type=float, default=-105.0)
    p_pred.add_argument("--alt", type=float, default=0.0)
    p_pred.add_argument("--hours", type=float, default=96.0)
    p_pred.add_argument("--min-el", type=float, default=10.0)
    p_pred.set_defaults(func=cmd_predict)

    p_dec = sub.add_parser("decode", help="decode SSTV images from WAV file(s)")
    p_dec.add_argument("files", nargs="+")
    p_dec.set_defaults(func=cmd_decode)

    p_demo = sub.add_parser("demo", help="headless simulated pass demo")
    p_demo.add_argument("--snr", type=float, default=15.0)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if not getattr(args, "cmd", None):
        args.cmd = "gui"
        args.simulate = False
        args.func = cmd_gui

    setup_logging()
    ensure_dirs()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

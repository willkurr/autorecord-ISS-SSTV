# ISS SSTV Station

An all-in-one amateur radio application for the International Space
Station's slow-scan television (SSTV) downlink: it tracks the ISS from
AMSAT's daily Keplerian elements, predicts passes over your location,
shows the satellite moving across your sky, records the pass from an
RTL-SDR with automatic doppler correction, decodes **Robot 36** images
live as they come down, and files everything away for later review.

```
┌────────────────────────────────────────────────────────────────────┐
│  Passes tab          Live pass tab              Images tab         │
│  ─ ground station    ─ az/el polar sky map      ─ image gallery    │
│  ─ pass predictions  ─ doppler/freq readouts    ─ metadata         │
│  ─ auto-record arm   ─ audio waterfall          ─ re-analyze WAVs  │
│                      ─ live SSTV decode cards                       │
└────────────────────────────────────────────────────────────────────┘
```

## Features

- **ISS tracking** – downloads the latest elements from AMSAT
  (`https://www.amsat.org/tle/dailytle.txt`, ISS = catalog 25544), caches
  them on disk and works offline from the cache.  Pass predictions list
  AOS/LOS, duration, peak elevation and pass direction for your station.
- **Az/El sky map** – polar view (N up, E right) with the full pass track,
  the travelled portion highlighted and the live position marked.
- **Software doppler correction** – the dongle stays parked at a fixed
  frequency; a per-sample numerically-controlled oscillator removes the
  predicted doppler shift (up to ±3.6 kHz at 145.8 MHz) glitch-free,
  which matters for SSTV.  The effective tuned frequency is displayed
  live.
- **Pass recording** – the doppler-corrected FM audio of the whole pass is
  written to a 48 kHz WAV as it arrives, with a JSON sidecar holding the
  TLE used, station coordinates, radio settings and the images decoded.
  Recording starts automatically at AOS (minus pre-roll) and stops after
  LOS when a pass is armed.
- **Live Robot 36 decoding** – a VIS-code detector watches the audio; when
  an image header appears, a decode card opens and the picture is rendered
  line by line.  Completed images are saved as PNGs and appear in the
  browser.  The mode registry makes adding other SSTV modes
  (Martin, Scottie, PD...) straightforward.
- **Image browser** – gallery of all decoded images with metadata, export,
  deletion, and **offline re-analysis**: point it at any saved pass
  recording (or any SSTV WAV) and decode it again.
- **No hardware? no problem** – a built-in pass simulator generates a
  realistic ISS pass (real doppler from the TLE, FM modulation, noise,
  SSTV images) so the whole application can be exercised without an
  RTL-SDR.  A sound-card input is also available for decoding from any
  receiver.

## Installation

### Requirements

This project requires Python 3.8+ (3.10+ recommended) and the following pip packages:

- numpy, scipy, skyfield, requests, PySide6
- For RTL-SDR reception: `pyrtlsdr` and the librtlsdr native library
- For audio monitoring / sound-card input: `sounddevice` + PortAudio

First, install all the required packages with

```bash
pip install -r requirements.txt
```

(Note that you may want to create a Python virtual environment first with `python -m venv .venv`, then run `.\.venv\Scripts\activate` (Windows) or `source ./.venv/bin/activate` (Mac/Linux) to activate the virtual environment. You now have a clean Python environment where the required packages can be installed with `pip` in the command above.)

### RTL-SDR driver

If using an RTL-SDR to capture satellite passes, the drivers can be installed with the following commands:

| OS | librtlsdr |
|----|-----------|
| Linux (Debian/Ubuntu) | `sudo apt install librtlsdr-dev` (package `rtl-sdr` for the CLI tools) |
| macOS | `brew install librtlsdr` |
| Windows | Install the [Zadig](https://zadig.akeo.ie/) WinUSB driver for the dongle, then place `rtlsdr.dll` on `PATH` or next to the program |

On Linux, you will probably need to blacklist the DVB-TV drivers (`dvb_usb_rtl28xxu`) if the dongle is claimed by the kernel:

```bash
echo "blacklist dvb_usb_rtl28xxu" | sudo tee /etc/modprobe.d/blacklist-rtl.conf
```

## Usage

Once you have the Python environment setup and packages/drivers have been installed, simply run `python main.py`.

Other run options are available:

```bash
python main.py                # launch the GUI
python main.py --simulate     # GUI with the pass simulator (no hardware)
python main.py predict        # CLI pass predictions for the next 4 days
python main.py decode pass.wav          # decode images from a recording
python main.py demo           # headless simulated pass, decoded end-to-end
```

### Receiving an ISS SSTV event

1. **Passes tab** – enter your latitude/longitude (and altitude), set the
   minimum useful elevation (10-15 deg works well) and predict passes.
2. Click **Track & record selected pass** – the pass is armed, the sky map
   shows its track, and with *Auto-record* enabled the receiver starts by
   itself before AOS.
3. **Settings → Radio** – set the downlink frequency (ISS SSTV: 
   **145.800 MHz FM**), your dongle's PPM error if known, and the gain.
4. When the pass begins the waterfall shows the signal, doppler is
   corrected continuously, and any SSTV transmission opens a live decode
   card.  Images land in the **Images** tab automatically.
5. After LOS the WAV recording and its JSON metadata are in
   `data/recordings/`; decoded images are in `data/images/`.

### Calibrating frequency error

The RTL-SDR's crystal error (PPM) shifts everything.  Set your dongle's
PPM in Settings → Radio (measure it against a known-strong station), and
use *Fine offset* for sub-PPM residuals: a 1 ppm error at 145.8 MHz is
146 Hz, which would tint decoded images by ~46 grey levels.

### Troubleshooting the dongle

Use **Settings → Source → Check RTL-SDR connection** to verify the whole
driver stack.  If the check fails, the dialog explains what to install
for your platform (Zadig driver + `rtlsdr.dll` on Windows,
`brew install librtlsdr` on macOS, `librtlsdr-dev` / `rtl-sdr-devel` /
`rtl-sdr` on Linux) and lists common fixes (dongle claimed by the DVB-TV
kernel driver, device in use by another program).

## Data layout

```
data/
├── settings.json        application settings
├── tle/dailytle.txt     cached AMSAT element set
├── recordings/          pass WAVs + JSON sidecars
└── images/              decoded images + index.json
```

Set `ISSSSTV_DATA_DIR` to relocate the data directory.

## How it works

```
RTL-SDR IQ (240 kS/s, fixed tune)
   │  complex NCO: removes tuning offset + doppler + fine error
   │  15 kHz channel FIR → ÷5 → FM discriminator → DC block → de-emphasis
   ▼
48 kHz FM audio ──┬─► WAV recorder (+ JSON sidecar)
                  ├─► spectrum/waterfall
                  ├─► audio monitor (optional)
                  └─► SSTV engine
                        ├─ band-pass 950-2450 Hz → analytic signal →
                        │  instantaneous frequency
                        ├─ VIS header detector (leader → start bit →
                        │  8 FSK bits → parity → stop → first sync check)
                        └─ Robot 36 line decoder (per-line sync re-anchoring,
                           Y + interleaved Cb/Cr → RGB), rendered live
```

Doppler is applied in software rather than by retuning the dongle:
hardware retunes inject phase discontinuities that corrupt SSTV timing,
while the NCO is sample-accurate and continuous.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers TLE parsing/caching, pass prediction and doppler against
real elements, a full Robot 36 encode→decode round trip (clean, noisy,
truncated, corrupt-parity, pathological chunking), the RF chain with
doppler, an end-to-end simulated pass, recorder round-trips and GUI smoke
tests (offscreen).

## Limitations & roadmap

- Only Robot 36 is decoded; the mode registry is ready for more modes.
- The RTL-SDR is the only hardware source; SoapySDR support would be a
  natural extension.
- PPM calibration is manual; auto-calibration from the 1200 Hz sync
  pulses is possible.
- Recording re-analysis currently decodes whole files; bookmarks/skip
  could make browsing long passes faster.

73!

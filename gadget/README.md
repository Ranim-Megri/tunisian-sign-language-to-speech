# Gadget Runtime (`gadget`) — scaffold

This folder is intended to hold the **Raspberry Pi 4 runtime** that ties together camera, OLED, speaker, buttons and LED with the sign classifier and TTS engine.

**Current state:** [`main.py`](main.py) contains a single placeholder comment. There is no runnable gadget program yet. The PC equivalent, [`main_pipeline.py`](../main_pipeline.py), is fully implemented.

---

## Intended responsibilities

| Step | Intended implementation |
|---|---|
| Camera | `picamera2` with the Pi Camera Module 3 (IMX708) |
| Hand detection | MediaPipe `HandLandmarker`, `assets/sign_data/hand_landmarker.task` |
| Features | `models/sign_classifier/training/features.py` → 129 floats |
| Classification | `models/sign_classifier/models/model.joblib` (24 classes) |
| Voting / sentence buffer | Mirror `main_pipeline.py` (8-frame buffer, 4 votes, 0.55 confidence, 1.2 s pause) |
| Display | 128×64 I2C OLED at `0x3C` via `luma.oled`, with `arabic_reshaper` / `python-bidi` |
| Speech | `tts.engine.TunisianTTSEngine` via the MAX98357A I2S amplifier |
| Inputs | Mute/reset buttons and status LED via `gpiozero` |

---

## Hardware mapping

Pins are documented in [`docs/hardware.md`](../docs/hardware.md); [`config.py`](../config.py) is a placeholder and does not yet define constants.

| Component | GPIO / interface |
|---|---|
| Mute button | GPIO 17 |
| Reset button | GPIO 27 |
| Status LED | GPIO 22 (through 330 Ω) |
| OLED | I2C, SDA GPIO 2, SCL GPIO 3, address `0x3C` |
| I2S amplifier | BCLK GPIO 18, LRC GPIO 19, DIN GPIO 21 |
| Camera | CSI port |

An interactive wiring guide is available at [`docs/phone-gadget-3d.html`](../docs/phone-gadget-3d.html).

---

## Environment preparation (for when the runtime is implemented)

```bash
sudo apt update && sudo apt install -y python3-picamera2 i2c-tools libasound2-dev
sudo raspi-config   # Interface Options → I2C → enable
pip install -r requirements.txt
pip install -r models/tts/requirements_tts.txt
```

`requirements.txt` already lists `luma.oled`, `gpiozero`, `arabic-reshaper`, `python-bidi` and other Pi-side dependencies.

---

## Next steps

1. Implement and run the individual checks in [`hardware_tests/`](../hardware_tests/README.md).
2. Move the pin numbers from `docs/hardware.md` into `config.py`.
3. Implement `main.py` as a state machine reusing `features.py`, `model.joblib` and `TunisianTTSEngine` rather than duplicating their logic.
4. Validate on the Pi camera, then consider a systemd service for auto-start.

---

## Reference links

- [`../README.md`](../README.md) — architecture and PC quick start
- [`../hardware_tests/README.md`](../hardware_tests/README.md) — hardware-check scaffolds
- [`../models/tts/README.md`](../models/tts/README.md) — TTS engine
- [`../docs/hardware.md`](../docs/hardware.md) — wiring and pinout

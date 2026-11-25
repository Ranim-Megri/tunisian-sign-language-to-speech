# Hardware Tests (`hardware_tests`) — scaffolds

Standalone scripts intended to verify each Raspberry Pi component **before** the gadget runtime is used.

**Current state:** every script contains only a placeholder comment; none of them performs a test yet. See [`gadget/README.md`](../gadget/README.md) for the runtime status and [`docs/hardware.md`](../docs/hardware.md) for wiring.

---

## Intended checks

| Script | Intended behaviour | Success indicator |
|---|---|---|
| `test_camera.py` | Find the Pi Camera (IMX708) with `picamera2` and capture one frame | Frame captured and saved |
| `test_oled.py` | Detect the 128×64 I2C OLED at `0x3C` and draw a test pattern | Pattern visible on display |
| `test_audio.py` | Play a known WAV (e.g. `assets/audio/3aslema.wav`) through the I2S amplifier | Spoken word heard |
| `test_buttons.py` | Read mute (GPIO 17) and reset (GPIO 27) button states | State changes when pressed |
| `test_led.py` | Toggle the green LED on GPIO 22 | LED blinks |

---

## Environment preparation

```bash
sudo apt update && sudo apt install -y python3-picamera2 i2c-tools libasound2-dev
sudo raspi-config   # Interface Options → I2C → enable
pip install -r requirements.txt
```

`config.py` at the project root is a placeholder; pin assignments currently live in `docs/hardware.md`.

---

## Future run commands

Once the scripts are implemented:

```bash
python -u hardware_tests/test_camera.py
python -u hardware_tests/test_oled.py
python -u hardware_tests/test_audio.py
python -u hardware_tests/test_buttons.py
python -u hardware_tests/test_led.py
```

The `-u` flag forces unbuffered output.

---

## Implementation checklist

- [ ] Camera capture via `picamera2`
- [ ] OLED detection via I2C (`i2cdetect -y 1` should show `3c`) and `luma.oled`
- [ ] I2S playback via `aplay` / `sounddevice`
- [ ] Button reads via `gpiozero`
- [ ] LED blink via `gpiozero`
- [ ] Pin numbers centralised in `config.py`

Wiring reference: [`docs/hardware.md`](../docs/hardware.md) and [`docs/phone-gadget-3d.html`](../docs/phone-gadget-3d.html).

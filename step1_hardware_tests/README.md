# Step 1: Hardware tests

## Goal

Check each part alone before any machine learning.

Test in this order: camera, OLED, audio, buttons, LED, fan.

## Before you start

Use the Raspberry Pi with power.
Connect one part at a time when possible.
Enable I2C with `sudo raspi-config`, then choose Interface Options and I2C.
Enable the audio amplifier I2S by following its vendor guide.

## What to do

1. Camera: run `rpicam-hello --list-cameras`. It should show `imx708`.
2. OLED: run `sudo i2cdetect -y 1`. It should show `3c`.
3. Audio: run `aplay -l` and check that it lists a sound card. Then play a test WAV file with `aplay`.
4. Buttons: use the test script. It will print a result later.
5. LED: use the test script. It will blink later.
6. Fan: power the Pi. The fan should spin. After one hour, `vcgencmd get_throttled` should show `throttled=0x0`.

## Commands

Run these commands later:

`sudo raspi-config`

`rpicam-hello --list-cameras`

`sudo i2cdetect -y 1`

`aplay -l`

`aplay path/to/test.wav`

`python3 test_buttons.py`

`python3 test_led.py`

`vcgencmd get_throttled`

## What you should see

- Camera: `rpicam-hello --list-cameras` shows `imx708`.
- OLED: `sudo i2cdetect -y 1` shows `3c`.
- Audio: `aplay -l` lists a sound card, and the test WAV file plays.
- Buttons and LED: the test scripts print or blink. The scripts will be written later.
- Fan: it spins when the Pi has power. After one hour, `vcgencmd get_throttled` shows `throttled=0x0`.

## If it does not work

- OLED blank: swap SCL and SDA. Check that the module is I2C. Check which driver it uses: SSD1306 or SSD1309.
- No sound: check that BCLK and LRC are not swapped. Check that VIN is on 5 V.
- Camera not listed: reseat the ribbon. The contacts face the pins inside the connector.
- Random reboots: use a real 5.1 V 3 A power supply.

## Checklist

- [ ] Camera checked
- [ ] OLED checked
- [ ] Audio checked
- [ ] Buttons checked
- [ ] LED checked
- [ ] Fan checked

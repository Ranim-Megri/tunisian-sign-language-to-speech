# Hardware

## Hardware list

- Raspberry Pi 4
- Camera Module 3 on the CSI port
- OLED 128x64 I2C address 0x3C
- MAX98357A amplifier
- 3 W 4 ohm speaker, 40x20 mm
- 2 push buttons
- Green LED with 330 ohm resistor
- 25 mm 5 V fan
- 5.1 V 3 A USB-C charger
- Phone-shaped case with inner space 66 x 172 x 23 mm

## Pin table

| Part | Wire | Physical pin | Name |
|---|---|---|---|
| OLED | GND | 9 | GND |
| OLED | VCC | 1 | 3.3 V |
| OLED | SCL | 5 | GPIO3 |
| OLED | SDA | 3 | GPIO2 |
| Amplifier | VIN | 4 | 5 V |
| Amplifier | GND | 14 | GND |
| Amplifier | BCLK | 12 | GPIO18 |
| Amplifier | LRC | 35 | GPIO19 |
| Amplifier | DIN | 40 | GPIO21 |
| Amplifier | GAIN and SD | empty | leave empty |
| Mute button | signal | 11 | GPIO17 |
| Reset button | signal | 13 | GPIO27 |
| Mute and reset buttons | shared ground | 6 | GND |
| LED | signal through 330 ohm resistor to long leg | 15 | GPIO22 |
| LED | short leg | 6 | GND |
| Fan | red | 2 | 5 V |
| Fan | black | 20 | GND |

## Assembly order

1. [ ] Mount the Pi.
2. [ ] Connect power.
3. [ ] Connect the camera.
4. [ ] Connect the display.
5. [ ] Connect the audio.
6. [ ] Connect the buttons.
7. [ ] Connect the LED.
8. [ ] Connect the fan.

## Safety

Power off before changing wires.
Put the OLED on 3.3 V.
Put the amplifier on 5 V.
Connect speaker wires only to the amplifier.

The interactive 3D wiring tutorial (`phone-gadget-3d.html`) must be copied into this folder.

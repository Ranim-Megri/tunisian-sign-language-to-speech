# Step 4: Gadget

## Goal

Run everything on the Pi.

## Before you start

Finish the hardware tests.
Create `data/landmarks.csv`.
Create `model.joblib`.
Check the audio file names in `assets/`.

## What to do

1. Take a picture.
2. Find hand landmarks.
3. Ask the model for a prediction.
4. Keep predictions from the last 10 pictures.
5. Accept a word when 8 votes agree.
6. Use a confidence value of 0.85.
7. Hold a sign for 0.5 seconds.
8. If there is no sign for 1.5 seconds, finish the sentence.
9. Show the sentence on the OLED.
10. Play the audio file.

The loop is: picture, landmarks, prediction, vote over the last 10, accept a word, pause of 1.5 seconds ends the sentence, show on the OLED, play audio.

Default values:

- Vote window: 10
- Votes needed: 8
- Confidence: 0.85
- Hold time: 0.5 seconds
- Sentence pause: 1.5 seconds

The mute button on GPIO17 turns the sound on or off.
The reset button on GPIO27 clears the current words.
The LED on GPIO22 is on while the camera works.

## Commands

Run this command later:

`python3 main.py`

## What you should see

The camera works while the LED is on.
Accepted words collect into a sentence.
After 1.5 seconds without a sign, the sentence appears on the OLED and its audio plays.
The mute and reset buttons work.

## If it does not work

Check Step 1 again.
Check the model file and audio file names.
Check the GPIO numbers.
Check the vote and pause values.
Write down every problem before changing a setting.

## Checklist

- [ ] Camera loop works
- [ ] Landmarks work
- [ ] Predictions work
- [ ] Voting works
- [ ] Words are accepted
- [ ] Sentence pause works
- [ ] OLED shows the sentence
- [ ] Audio plays
- [ ] Mute button works
- [ ] Reset button works
- [ ] LED shows camera activity

## Later

- Show Arabic text on the OLED.
- Match similar sentences.
- Start automatically at boot.

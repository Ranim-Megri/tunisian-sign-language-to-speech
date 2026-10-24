# tunsl-gadget

This project is a small gadget for Tunisian Sign Language (TunSL).
It uses a camera to recognize still hand shapes.
It builds a short sentence.
It shows the sentence on an OLED screen.
It plays an audio clip in Tunisian Derja.
It runs on a Raspberry Pi 4.

We are a team of four beginners.
We will learn as we go.
We have six weeks.

## How it works

1. The camera takes a picture.
2. MediaPipe finds the points of the hand (landmarks).
3. A small scikit-learn model says which sign it is.
4. If the same sign is seen in most of the last 10 pictures, the word is accepted.
5. If no sign is seen for 1.5 seconds, the sentence is finished: it is shown on the OLED and an audio file is played.

## Folders and files

- `assets/` stores sign and sentence data, plus audio placeholders.
- `data/` stores datasets and is not uploaded to git.
- `step1_hardware_tests/` holds simple hardware test placeholders.
- `step2_explore_data/` holds dataset exploration placeholders.
- `step3_train/` holds model training placeholders.
- `step4_gadget/` holds the final gadget placeholder.
- `docs/` stores hardware, learning, and team documents.
- `requirements.txt` lists the Python packages.
- `config.py` will hold pins and settings later.
- `.gitignore` lists files that git should ignore.

## Steps

### Step 1: Hardware tests

Status: Not started

Check the camera, OLED, audio, buttons, LED, and fan.
See `step1_hardware_tests/README.md`.

### Step 2: Explore data

Status: Not started

Understand the pictures and extract hand landmarks.
See `step2_explore_data/README.md`.

### Step 3: Train

Status: Not started

Train and check a small model.
See `step3_train/README.md`.

### Step 4: Gadget

Status: Not started

Run the complete gadget on the Raspberry Pi.
See `step4_gadget/README.md`.

## Team roles

- A: machine learning and data
- B: hardware
- C: software integration
- D: content, audio and testing

## Limits

- The first version supports only still signs.
- The sentences are not yet validated by a Deaf signer.
- The dataset pictures come from other cameras, so we need our own pictures with the gadget camera.
- Accuracy numbers on the dataset are optimistic.

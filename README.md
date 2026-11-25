# Tunisian Sign Language to Speech

Recognises **Tunisian Sign Language (TunSL) hand shapes** from a camera or a single image, turns them into Tunisian Derja phrases, and **speaks them aloud**. The PC pipeline, classifier and TTS engine are implemented; the Raspberry Pi gadget and hardware checks are currently scaffolds.

---

## Architecture / data flow

```
Webcam or image
    │
    ▼
MediaPipe HandLandmarker  (2 hands, 0.35 detection/presence/tracking confidence)
    │
    ▼
landmarks_to_features()   → 129 normalised floats (21 × 3 per hand + hand offset)
    │
    ▼
models/sign_classifier/models/model.joblib
StandardScaler + RBF SVM (SVC, C=10, class_weight="balanced"), 24 sign classes
    │
    ▼
Vote buffer (8 frames, commit at 4 matching votes, confidence ≥ 0.55)
    │
    ▼
Sentence buffer  →  immediate auto-speech on commit; composite phrase after 1.2 s pause
    │
    ▼
TunisianTTSEngine.synthesize()
cached clip → natural Derja phrase → Edge-TTS online → word stitching → fallback clip
    │
    ▼
results/tts_output.wav  →  AudioPlayer.play_wav(..., async_play=True)
```

The live pipeline is implemented in [`main_pipeline.py`](main_pipeline.py). The classifier, training scripts and TTS package live under [`models/`](models/README.md).

---

## Implementation status

| Subsystem | Status | Where |
|---|---|---|
| PC webcam / image / simulation pipeline | Implemented | [`main_pipeline.py`](main_pipeline.py) |
| Feature extraction, training, evaluation | Implemented | [`models/sign_classifier/`](models/README.md) |
| Hybrid Derja TTS engine + CLI | Implemented | [`models/tts/`](models/tts/README.md) |
| TTS / classifier test tools | Implemented | [`tests/`](tests/) |
| Sign metadata, audio and TTS data | Present | [`assets/sign_data/`](assets/sign_data/README.md), [`assets/audio/`](assets/audio/README.md), [`assets/tts_data/`](assets/tts_data/README.md) |
| Raspberry Pi runtime | **Placeholder only** | [`gadget/main.py`](gadget/README.md) |
| Hardware verification scripts | **Placeholders only** | [`hardware_tests/`](hardware_tests/README.md) |
| GPIO settings module | **Placeholder only** | [`config.py`](config.py) |
| Wiring / assembly guide | Documented | [`docs/hardware.md`](docs/hardware.md), [`docs/phone-gadget-3d.html`](docs/phone-gadget-3d.html) |

---

## Repository layout

```
tunisian-sign-language-to-speech/
├── README.md                       # This file
├── config.py                       # Placeholder for GPIO pins/settings
├── main_pipeline.py                # PC pipeline: webcam / image / simulation → TTS
├── requirements.txt                # Core Python dependencies
│
├── models/                                  # Runtime modules + trained artefacts
│   ├── README.md
│   ├── hand_landmarker.task                 # Fallback MediaPipe task file
│   ├── sign_classifier/
│   │   ├── __init__.py
│   │   ├── exploration/                     # Dataset + landmark extraction notebooks
│   │   │   ├── explore_dataset.ipynb
│   │   │   └── extract_landmarks.ipynb
│   │   ├── inference/test_live.py           # Standalone webcam classifier test
│   │   ├── models/                          # model.joblib, landmarks.csv, metrics
│   │   ├── results/                         # Duplicate landmarks + failed images log
│   │   └── training/                        # features.py, compare_models.py, train.ipynb
│   └── tts/                                 # TTS package (imported as `tts`)
│       ├── __init__.py, engine.py, inference.py, text_normalizer.py
│       ├── audio_player.py, dataset.py, train.py
│       ├── requirements_tts.txt
│       └── README.md
│
├── assets/
│   ├── sign_data/                  # MediaPipe task, words.json, sentences.json, image dataset
│   ├── audio/                      # 95 mirrored WAV clips
│   ├── tts_data/                   # filelists/, raw/, wavs/ (TTS cache)
│   ├── words.json                  # Legacy root-level copy (fallback)
│   └── sentences.json              # Legacy root-level copy (fallback)
│
├── gadget/                         # Raspberry Pi runtime (scaffold)
├── hardware_tests/                 # Pi hardware checks (scaffolds)
├── tests/                          # Classifier + TTS test tools
├── results/                        # Generated audio (tts_output.wav, test_outputs/)
└── docs/                           # hardware.md, phone-gadget-3d.html, command reference
```

Older `step1_*` / `step2_*` / `step3_*` / `step4_*` folders were replaced by this layout.

---

## Recognised signs (24 classes)

| Category | Tokens |
|---|---|
| Requests / greetings | `3aslema`, `labes`, `oui`, `non`, `lyoum`, `enti`, `n3awnek` |
| Destinations | `baladya`, `banka`, `sbitar`, `dar`, `ma7kma` |
| Transport | `taxi`, `louage`, `car` (bus), `karhba` (car) |
| Days | `thnin`, `thleth`, `erb3a`, `5mis`, `jom3a`, `sebt`, `a7ad` |
| Family | `om` |

Signing `sbitar` then `taxi` produces → **"نحب نمشي للسبيطار بالتاكسي"** (I want to go to the hospital by taxi). Metadata for 46 sentence templates is in [`assets/sign_data/sentences.json`](assets/sign_data/sentences.json).

---

## Quick start (PC)

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r models/tts/requirements_tts.txt
```

`picamera2` is installed on the Pi with `sudo apt install python3-picamera2`, not through pip.

### 2. Provide the MediaPipe hand model

The pipeline expects `assets/sign_data/hand_landmarker.task`. If it is missing, download it:

```bash
curl -o assets/sign_data/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

See [`models/README.md`](models/README.md) for the fallback search paths.

### 3. Run

```bash
python main_pipeline.py                              # live webcam
python main_pipeline.py --image path/to/sign.jpg     # single image
python main_pipeline.py --test-signs sbitar taxi     # no camera, TTS only
python main_pipeline.py --camera 1 --confidence 0.6 --no-mirror
```

Live-window controls: **S** toggle auto-voice · **M** mirror · **SPACE** speak now · **C** clear · **Q** quit.

### 4. Other entry points

```bash
# Standalone webcam classifier (no TTS); auto-downloads the task file if needed
python models/sign_classifier/inference/test_live.py --model model.joblib

# Model info / single image / live classifier
python tests/run_sign_classifier.py --info
python tests/run_sign_classifier.py --image "assets/sign_data/(First ever) Tunisian Sign Language Dataset/Data/Destinations/sbitar/sbitar (1).jpg"

# TTS only
python -m models.tts.inference --words sbitar taxi --play
python tests/run_tts.py --list-phrases
python tests/test_interactive_phrases.py --interactive

# Tests (some TTS inputs use the online Edge-TTS fallback)
python -m pytest tests/ -v
```

---

## Classifier training and results

Training data: `models/sign_classifier/models/landmarks.csv` — **1302 rows**, **24 signs**, **129 features** per row. `compare_models.py` evaluates six classifiers on two splits (per-sign first-70%/last-30% "careful" split and a stratified random split), augments training data with horizontally flipped landmark vectors, then retrains the best model on all rows and saves it as `model.joblib`.

Careful-split results from `comparison_metrics.csv`:

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| **rbf_svm** (current `model.joblib`) | **0.8925** | **0.8685** | **0.8763** |
| extra_trees | 0.8875 | 0.8598 | 0.8700 |
| random_forest | 0.8450 | 0.8131 | 0.8264 |
| hist_gradient_boosting | 0.8175 | 0.7977 | 0.7892 |
| logistic_regression | 0.8100 | 0.7895 | 0.7862 |
| knn | 0.8025 | 0.7737 | 0.7632 |

Retrain with:

```bash
python models/sign_classifier/training/compare_models.py
```

Details: [`models/sign_classifier/training/README.md`](models/sign_classifier/training/README.md).

---

## Raspberry Pi (scaffold)

The hardware is wired and documented, but the Pi software is not implemented yet:

- [`gadget/main.py`](gadget/README.md) contains only a placeholder comment.
- [`hardware_tests/*.py`](hardware_tests/README.md) contain only placeholder comments.
- [`config.py`](config.py) contains only a placeholder comment; pin numbers are documented in [`docs/hardware.md`](docs/hardware.md).

Hardware: Raspberry Pi 4, Camera Module 3, 128×64 I2C OLED (0x3C), MAX98357A I2S amplifier + 3 W speaker, mute/reset buttons, green LED, fan.

Dependencies for the intended Pi runtime (`luma.oled`, `gpiozero`, `arabic-reshaper`, `python-bidi`, etc.) are already listed in `requirements.txt`.

---

## Arabizi notation

| Symbol | Arabic | Sound |
|---|---|---|
| `3` | ع | deep throat "a" |
| `7` | ح | strong "h" |
| `5` | خ | "kh" |
| `9` | ق | deep "k" |
| `8` | غ | "gh" |
| `2` | ء | hamza |

`text_normalizer.py` maps these tokens to Arabic script for TTS. Example: `3aslema` = عسلامة · `sbitar` = سبيطار · `5mis` = الخميس.

---

## Known limitations

- Only **static hand poses** from individual frames are classified; dynamic gestures and sign sequences are not modelled.
- Offline TTS covers 95 clips; unseen text falls back to **Edge-TTS** (needs internet) or simple word stitching.
- The MediaPipe task file is not auto-downloaded by `main_pipeline.py` (it is by `test_live.py`).
- The Raspberry Pi runtime, hardware tests and `config.py` are placeholders, not runnable programs.
- The image dataset contains 60 sign folders / 4423 images, but the current classifier is trained on the 24 verified classes only.
- Accuracy was measured on webcam-derived landmark data; Pi camera performance is unverified.

---

## Documentation index

- [`models/README.md`](models/README.md) — classifier artefacts and MediaPipe paths
- [`models/tts/README.md`](models/tts/README.md) — TTS engine, caches and CLI
- [`models/sign_classifier/training/README.md`](models/sign_classifier/training/README.md) — features and evaluation
- [`assets/sign_data/README.md`](assets/sign_data/README.md) — signs, phrases and image dataset
- [`assets/audio/README.md`](assets/audio/README.md) — mirrored WAV clips
- [`assets/tts_data/README.md`](assets/tts_data/README.md) — TTS dataset and filelists
- [`gadget/README.md`](gadget/README.md) — Raspberry Pi runtime scaffold
- [`hardware_tests/README.md`](hardware_tests/README.md) — hardware-check scaffolds
- [`docs/hardware.md`](docs/hardware.md) — wiring and pinout

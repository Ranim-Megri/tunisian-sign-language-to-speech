# Tunisian TTS Dataset (`assets/tts_data`)

Pre-rendered audio clips and transcription filelists used by [`models/tts/engine.py`](../../models/tts/README.md).

---

## Layout

```
assets/tts_data/
├── filelists/
│   ├── all.txt       # 20 free-form Derja sentences (tn_0001 … tn_0020)
│   ├── tn_train.txt  # 17 lines
│   ├── tn_val.txt    # 2 lines
│   └── tn_test.txt   # 1 line
├── raw/              # 31 original recordings (kept for reference)
├── wavs/             # 95 normalised, silence-trimmed clips used at runtime
│   ├── 24 single-word clips (3aslema.wav … karhba.wav)
│   ├── 41 two-word combos (sbitar_taxi.wav … thnin_sbitar.wav)
│   ├── 10 sentence clips (s01.wav … s10.wav)
│   └── 20 free-form clips (tn_0001.wav … tn_0020.wav)
└── README.md
```

`wavs/` is mirrored in [`assets/audio/`](../audio/README.md).

---

## How the engine uses these files

At startup `TunisianTTSEngine._build_fast_cache()`:

1. Reads `filelists/all.txt`, registering each transcript (Arabizi and Arabic forms) → WAV in `fast_cache`.
2. Reads `assets/sign_data/sentences.json` and `words.json` for word-set and word caches.
3. Scans `wavs/*.wav`, registering each stem as a word and each underscore-separated stem as an order-independent combo.

Only 20 clips are indexed in `all.txt`; the other 75 become available through the JSON metadata and the directory scan.

---

## Filelist format

Pipe-delimited, LJSpeech-style:

```
assets/tts_data/wavs/tn_0005.wav|n7eb nemchi lel sbitar bel taxi
```

Left: WAV path relative to the project root. Right: Arabizi transcript. `models/tts/dataset.py` resolves paths against the project root, then `assets/tts_data/wavs/`, then `assets/audio/`.

---

## Audio specifications

| Property | Value |
|---|---|
| Format | RIFF PCM WAV |
| Sample rate | 22 050 Hz |
| Bit depth | 16-bit signed |
| Channels | Mono |
| Silence handling | Leading/trailing silence trimmed during generation |

---

## Dataset origin

1. A vocabulary of 24 sign words, 41 two-word combinations, 10 sentence clips and 20 free-form Derja sentences was assembled.
2. Clips were synthesised with **edge-tts** using the `ar-TN-HediNeural` voice, resampled to 22 050 Hz and silence-trimmed.
3. Files were written to `wavs/` and indexed in `filelists/*.txt` and `assets/sign_data/sentences.json`.

---

## Arabizi notation

| Symbol | Arabic | Sound |
|---|---|---|
| `3` | ع | deep throat "a" |
| `7` | ح | strong "h" |
| `5` | خ | "kh" |
| `9` | ق | deep "k" |
| `8` | غ | "gh" |

---

## Adding clips

1. Create a 22 050 Hz, 16-bit mono WAV and place it in `wavs/`.
2. Append a line to `filelists/all.txt` (or to the appropriate train/val/test split).
3. Mirror the file in [`assets/audio/`](../audio/README.md).
4. For sign-token lookup, add a template to [`assets/sign_data/sentences.json`](../sign_data/README.md).
5. Validate with:
   ```bash
   python -m models.tts.train --filelist assets/tts_data/filelists/all.txt
   ```

Note: `models/tts/train.py` only validates the dataset — there is no `--add` helper.

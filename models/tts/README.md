# Tunisian Derja TTS (`models/tts`)

Hybrid text-to-speech package for Tunisian Derja and Arabizi. It serves pre-rendered clips instantly, builds natural Derja sentences from sign tokens, and falls back to online Edge-TTS or local word stitching.

Import name is `tts` (because `models/` is on `sys.path` in the entry scripts).

---

## Package layout

| Module | Purpose |
|---|---|
| `__init__.py` | Exports `TunisianTTSEngine`, `synthesize`, `speak`, `get_engine`, `AudioPlayer`, normalisation helpers and `TunisianTTSDataset`. |
| `engine.py` | `TunisianTTSEngine`: builds caches, selects a synthesis path, generates online speech, stitches word clips, writes the output WAV. |
| `inference.py` | Module-level `synthesize()` / `speak()` wrappers and a CLI (`--text`, `--words`, `--out`, `--play`, `--list-phrases`). |
| `text_normalizer.py` | Arabizi → Arabic mapping (`3`→ع, `7`→ح, `5`→خ, `9`→ق, …), Derja dictionary, text cleaning, character vocabulary. |
| `audio_player.py` | Cross-platform playback: `aplay`/`paplay`/`play` on Linux, `winsound`/PowerShell on Windows, `afplay` on macOS, `sounddevice` as fallback. |
| `dataset.py` | Reads pipe-delimited filelists and validates/loads audio samples. |
| `train.py` | **Dataset validation only** — prints entry counts, duration and a sample; it does not train a model. |
| `requirements_tts.txt` | `edge-tts`, `soundfile`, `numpy`, `scipy`. |

---

## Engine initialisation

```python
from tts.engine import TunisianTTSEngine          # or: from models.tts.engine import TunisianTTSEngine
engine = TunisianTTSEngine()                       # voice="ar-TN-HediNeural", sample_rate=22050
```

`_build_fast_cache()` reads three sources from [`assets/tts_data/`](../../assets/tts_data/README.md) and [`assets/sign_data/`](../../assets/sign_data/README.md):

| Source | What it contributes |
|---|---|
| `assets/tts_data/filelists/all.txt` (20 entries) | Full free-form Derja sentences (`tn_0001.wav` … `tn_0020.wav`), registered under Arabizi and Arabic keys. |
| `assets/sign_data/sentences.json` (46 entries) | Word-set → clip mappings (`combo_cache`) and sentence text keys. |
| `assets/sign_data/words.json` (24 entries) | Single-word clips (`word_cache`). |
| `assets/tts_data/wavs/*.wav` (95 files) | Every clip stem registered as a word; underscore-separated stems registered as order-independent combos. |

Fallback locations (`assets/words.json`, `assets/sentences.json`, `assets/audio/`) are checked only if the primary files are absent.

Three caches are built:

| Cache | Key | Value |
|---|---|---|
| `fast_cache` | Normalised text / Arabizi / Arabic string | `Path` to WAV |
| `word_cache` | Single token | `Path` to WAV |
| `combo_cache` | `frozenset` of tokens (order-independent) | `Path` to WAV |

---

## Synthesis order — `synthesize(text_or_tokens, output_path=None)`

1. **`find_cached_audio()`** — match token set in `combo_cache`, then normalised/exact/partial text in `fast_cache`.
2. **Natural Derja grammar** — `_build_natural_derja()` assembles phrases such as `نحب نمشي للسبيطار بالتاكسي` for destination+transport tokens, day+destination, family+destination and greeting pairs; retry the caches with that text.
3. **Online Edge-TTS** — `edge_tts.Communicate(text, "ar-TN-HediNeural")`, 3.5 s timeout; MP3 → 22 050 Hz mono PCM, silence trimmed; result cached.
4. **Local word stitching** — concatenate cached word clips with **60 ms silence** and a **5 ms fade** at both ends.
5. **Fallback** — `assets/tts_data/wavs/tn_0001.wav`, else 0.5 s of silence.

Output defaults to `results/tts_output.wav`. The return value is `(path, audio_array, sample_rate)`.

`_build_natural_derja()` vocabulary: destinations `sbitar, baladya, banka, dar, ma7kma`; transports `taxi, louage, car, karhba`; days `thnin … a7ad, lyoum`; greetings `3aslema, labes, oui, non, enti, n3awnek, om`.

Playback: `speak(text, async_playback=True)` or `player.play_wav(path, async_play=True)` (the pipeline uses async playback to keep the UI responsive).

---

## How the PC pipeline calls it

```python
# main_pipeline.py
from tts.engine import TunisianTTSEngine
self.tts = TunisianTTSEngine()

out_wav, _, _ = self.tts.synthesize(words_to_speak)   # e.g. ["sbitar", "taxi"]
self.tts.player.play_wav(out_wav, async_play=True)
```

---

## CLI usage

```bash
# Speak sign tokens (cached combos are order-independent)
python -m models.tts.inference --words sbitar taxi --play

# Speak free-form Arabizi or Arabic text
python -m models.tts.inference --text "3aslema chnowa 7alek lyoum ?" --play

# Choose output path, list cached phrases (first 40)
python -m models.tts.inference --words sbitar taxi --out results/demo.wav
python -m models.tts.inference --list-phrases

# Show every cached phrase / interactive tester
python tests/run_tts.py --list-phrases
python tests/test_interactive_phrases.py --interactive
```

Dataset validation:

```bash
python -m models.tts.train --filelist assets/tts_data/filelists/all.txt
```

---

## Audio coverage (95 clips in `assets/tts_data/wavs/`)

| Kind | Count | Examples |
|---|---|---|
| Single sign words | 24 | `3aslema.wav`, `sbitar.wav`, `taxi.wav` |
| Named two-word combinations | 41 | `sbitar_taxi.wav`, `thnin_baladya.wav` |
| Numbered sentence clips | 10 | `s01.wav` … `s10.wav` (from `sentences.json`) |
| Free-form Derja clips | 20 | `tn_0001.wav` … `tn_0020.wav` (in `all.txt`) |

All clips are 22 050 Hz, 16-bit mono WAV.

---

## Adding new clips

1. Create a 22 050 Hz, 16-bit mono WAV.
2. Place it in `assets/tts_data/wavs/` (and mirror it in `assets/audio/` if the Pi runtime needs it).
3. Append a line to `assets/tts_data/filelists/all.txt`:
   ```
   assets/tts_data/wavs/my_phrase.wav|arabizi transcription
   ```
4. For sign-token lookup, add a matching entry to `assets/sign_data/sentences.json`.
5. Restart the process — caches are built at engine startup.

---

## Known limitations

- Unseen text requires internet for Edge-TTS; otherwise it is assembled from individual word clips (flat prosody).
- Combination lookup ignores token order: `["sbitar", "taxi"]` and `["taxi", "sbitar"]` resolve to the same clip.
- There is no neural TTS training in `train.py` — it only validates the dataset.
- `--list-phrases` in `models/tts/inference.py` prints only the first 40 keys; `tests/run_tts.py` prints all.

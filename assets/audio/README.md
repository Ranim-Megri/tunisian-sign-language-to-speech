# Audio Assets (`assets/audio`)

Mirror of the ready-to-use WAV clips in [`../tts_data/wavs/`](../tts_data/README.md). The TTS engine checks this directory as a fallback when a clip is not found in `assets/tts_data/wavs/`.

## Contents

- **95 WAV files**, 22 050 Hz, 16-bit signed PCM, mono.
- Single sign words (24), named two-word combinations (41), numbered sentence clips `s01.wav`…`s10.wav` (10), and free-form clips `tn_0001.wav`…`tn_0020.wav` (20).
- `README.md` — this file.

No metadata or MediaPipe model files are stored here; the canonical copies are in [`../sign_data/`](../sign_data/README.md). Code paths that look for `words.json`, `sentences.json` or `hand_landmarker.task` in this folder are fallbacks only.

## Related

- [`../tts_data/README.md`](../tts_data/README.md) — canonical TTS cache and filelists
- [`../../models/tts/README.md`](../../models/tts/README.md) — how clips are selected at runtime
- [`../../hardware_tests/README.md`](../../hardware_tests/README.md) — intended speaker check (`test_audio.py`, currently a placeholder)

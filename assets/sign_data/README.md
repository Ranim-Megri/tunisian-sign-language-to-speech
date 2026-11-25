# Sign Data Assets (`assets/sign_data`)

Sign metadata, the MediaPipe hand-landmark model, and the source image dataset used for training.

---

## Contents

| Item | Description |
|---|---|
| `hand_landmarker.task` | Pre-trained MediaPipe Hand-Landmarker (float16). **Primary copy** used by `main_pipeline.py` and `test_live.py`. The PC pipeline does not auto-download it. |
| `words.json` | **24** verified sign entries (tokens used by the classifier and TTS caches). |
| `sentences.json` | **46** verified sentence templates mapping word sets → audio clips / Derja text. |
| `(First ever) Tunisian Sign Language Dataset/` | Source image dataset: `LICENSE`, a reference PDF, and `Data/` with **4423 JPGs** in **60 sign folders** under 5 categories (`Demandes`, `Destinations`, `Famille`, `Jours`, `Transport`). |
| `README.md` | This file. |

`assets/words.json` and `assets/sentences.json` at the project root are legacy duplicates; the TTS engine checks them only as fallbacks.

---

## `hand_landmarker.task`

Download once if missing:

```bash
curl -o assets/sign_data/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

| Caller | Behaviour if missing |
|---|---|
| `main_pipeline.py` | Checks this path first, then `models/sign_classifier/models/`, `models/`, `assets/audio/`; raises `FileNotFoundError` if none exist. |
| `models/sign_classifier/inference/test_live.py` | Checks this path, then `models/hand_landmarker.task`; auto-downloads if neither exists. |
| `tests/run_sign_classifier.py` | Checks this path, then `models/hand_landmarker.task`. |

See [`models/README.md`](../../models/README.md) for the options used with the landmarker.

---

## `words.json` — 24 verified signs

Fields per entry: `category`, `folder` (= pipeline token), `derja`, `french`, `audio`, `status` (`"verified"`).

| Category | Tokens |
|---|---|
| Demandes | `3aslema`, `labes`, `oui`, `non`, `lyoum`, `enti`, `n3awnek` |
| Destinations | `baladya`, `banka`, `sbitar`, `dar`, `ma7kma` |
| Transport | `taxi`, `louage`, `car` (bus), `karhba` (car) |
| Jours | `thnin`, `thleth`, `erb3a`, `5mis`, `jom3a`, `sebt`, `a7ad` |
| Famille | `om` |

| Token | Derja | French |
|---|---|---|
| `3aslema` | عسلامة | Bonjour |
| `labes` | لاباس | Ça va ? |
| `oui` | إيه | Oui |
| `non` | لا | Non |
| `lyoum` | اليوم | Aujourd'hui |
| `enti` | إنت | Toi |
| `om` | ماما | Maman |
| `n3awnek` | نعاونك | T'aider |
| `thnin` | الإثنين | Lundi |
| `thleth` | الثلاثاء | Mardi |
| `erb3a` | الإربعاء | Mercredi |
| `5mis` | الخميس | Jeudi |
| `jom3a` | الجمعة | Vendredi |
| `sebt` | السبت | Samedi |
| `a7ad` | الأحد | Dimanche |
| `baladya` | بلدية | Mairie |
| `banka` | بنكة | Banque |
| `sbitar` | سبيطار | Hôpital |
| `dar` | دار | Maison |
| `ma7kma` | محكمة | Tribunal |
| `taxi` | تاكسي | Taxi |
| `louage` | لواج | Louage |
| `car` | كار | Bus |
| `karhba` | كرهبة | Voiture |

---

## `sentences.json` — 46 templates

Fields: `id`, `words` (sign tokens), `derja`, `french`, `audio`, `status`. The first ten use numbered clips `s01.wav`…`s10.wav`; the rest point at named clips such as `baladya_taxi.wav`.

Coverage: single words, destination + transport, day + destination, family + destination, and short dialogues. The TTS engine registers each `words` set as an **order-independent** combo key.

Example: `{"words": ["taxi", "sbitar"], "derja": "نحب نمشي للسبيطار بالتاكسي"}`.

---

## Adding a sign

1. Add images under `assets/sign_data/(First ever) Tunisian Sign Language Dataset/Data/<category>/<token>/`.
2. Run `models/sign_classifier/exploration/extract_landmarks.ipynb` to add rows to `landmarks.csv`, then retrain with `compare_models.py`.
3. Add an entry to `words.json` (set `status: "verified"` once audio exists).
4. Add a WAV to `assets/tts_data/wavs/` and mirror it in `assets/audio/`.
5. Add relevant `sentences.json` entries and update `assets/tts_data/filelists/all.txt`.

Related: [`assets/audio/README.md`](../audio/README.md), [`assets/tts_data/README.md`](../tts_data/README.md), [`models/sign_classifier/training/README.md`](../../models/sign_classifier/training/README.md).

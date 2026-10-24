# Assets

The asset files describe signs, sentences, and audio names.

## `words.json`

| Field | Meaning |
|---|---|
| `category` | The dataset category for the sign. |
| `folder` | The dataset folder name for the sign. |
| `derja` | The Tunisian Derja word in Arabic script. |
| `french` | A French description of the word. |
| `audio` | The audio file name that will be recorded later. |
| `status` | Use `to_verify` until a person who knows TunSL confirms the sign. |

To add a new sign, add one object to the list.
Copy the folder name exactly.
Add the Derja word, the French description, and the future audio file name.
Set `status` to `to_verify`.
Do not change this status until a person who knows TunSL confirms the sign.

## `sentences.json`

| Field | Meaning |
|---|---|
| `words` | A list of sign folder names in sentence order. |
| `derja` | The draft sentence in Tunisian Derja. |
| `french` | The draft French translation. |
| `audio` | The future audio file name. |
| `status` | The review state. Draft examples are not validated yet. |

## Audio

Record every audio file as WAV.
Use mono sound at 22.05 kHz.
Keep the volume similar for all files.
Use a Tunisian native speaker.
Record the files later in `assets/audio/`.

Every sign must keep `status: "to_verify"` until a person who knows TunSL confirms it.

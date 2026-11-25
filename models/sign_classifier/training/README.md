# Sign Classifier Training (`models/sign_classifier/training`)

Feature extraction and model selection for the 24-class TunSL hand-shape classifier.

---

## Files

| File | Purpose |
|---|---|
| `features.py` | `get_features(landmarker, img)` and `landmarks_to_features(hand_landmarks_list)` — both return a **129-float** vector, or `None` when no hand is found. Imported by `main_pipeline.py`, `test_live.py`, `tests/run_sign_classifier.py` and the notebooks. |
| `compare_models.py` | Loads `landmarks.csv`, evaluates six classifiers on two splits with horizontal-flip augmentation, retrains the winner on all rows and saves `model.joblib` + `comparison_metrics.csv`. |
| `train.ipynb` | Interactive version of the training/evaluation pipeline. |
| `README.md` | This file. |

---

## Feature extraction (`features.py`)

`landmarks_to_features()` is the function used at runtime:

- 21 MediaPipe landmarks per hand, each `(x, y, z)`.
- Coordinates are made **wrist-relative**, then divided by the distance from the wrist (landmark 0) to the middle-finger MCP (landmark 9), plus `1e-6`, giving scale invariance.
- Hands are sorted by wrist x-coordinate so hand order is stable.
- Layout of the 129 values:

| Indices | Content |
|---|---|
| `0:63` | First hand: 21 × 3 normalised coordinates |
| `63:126` | Second hand: 21 × 3 normalised coordinates (zeros if only one hand) |
| `126:129` | Wrist offset of hand 2 relative to hand 1 (zeros if only one hand) |

`get_features(landmarker, img)` runs MediaPipe detection on a BGR frame and delegates to `landmarks_to_features()`.

---

## Training data and evaluation

`models/sign_classifier/models/landmarks.csv` contains **1302 rows** with columns `sign`, `path` and `f0`…`f128`, covering **24 signs** (between 12 and 109 images per sign).

`compare_models.py` evaluates:

- **Careful split** — per sign sorted by `(sign, path)`, first 70% train / last 30% test.
- **Random split** — stratified 70/30 with `random_state=0`.
- Training rows are augmented with horizontally mirrored landmark vectors (x coordinates negated; hands swapped when two are present).
- Models: logistic regression, RBF SVM, ExtraTrees, RandomForest, KNN (5 neighbours), HistGradientBoosting. All use `random_state=0`.
- Scores: accuracy, balanced accuracy, macro F1. The best careful-split macro F1 wins and is retrained on all rows.

---

## Retrain

```bash
# from the project root
python models/sign_classifier/training/compare_models.py

# optional: point at another landmark CSV (falls back to results/ or root results/)
python models/sign_classifier/training/compare_models.py --data path/to/landmarks.csv
```

Outputs (always written to `models/sign_classifier/models/`):

- `model.joblib` — the best model, retrained on all augmented data.
- `comparison_metrics.csv` — both splits for all six models.

---

## Latest results (careful split, `comparison_metrics.csv`)

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| **rbf_svm** (selected) | **0.8925** | **0.8685** | **0.8763** |
| extra_trees | 0.8875 | 0.8598 | 0.8700 |
| random_forest | 0.8450 | 0.8131 | 0.8264 |
| hist_gradient_boosting | 0.8175 | 0.7977 | 0.7892 |
| logistic_regression | 0.8100 | 0.7895 | 0.7862 |
| knn | 0.8025 | 0.7737 | 0.7632 |

The current `model.joblib` is the `rbf_svm` pipeline: `StandardScaler()` → `SVC(C=10, gamma='scale', probability=True, class_weight='balanced', random_state=0)`.

---

## Checklist

- [ ] `models/sign_classifier/models/landmarks.csv` is up to date
- [ ] `compare_models.py` has been run and the metrics reviewed
- [ ] `models/sign_classifier/models/model.joblib` is the selected model
- [ ] `main_pipeline.py --model model.joblib` loads it successfully

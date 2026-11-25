"""Compare landmark classifiers and save models for live testing.

Run from the project root:
    python models/sign_classifier/training/compare_models.py
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = [f"f{i}" for i in range(129)]
RANDOM_STATE = 0


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_landmarks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as file:
        rows.extend(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"No landmark rows found in {path}")

    rows.sort(key=lambda row: (row["sign"], row["path"]))
    labels = np.array([row["sign"] for row in rows])
    features = np.array(
        [[float(row[name]) for name in FEATURES] for row in rows],
        dtype=float,
    )
    return features, labels


def flip_features(X_arr: np.ndarray) -> np.ndarray:
    """Horizontally mirror landmark feature vectors for data augmentation."""
    X_flip = np.zeros_like(X_arr)
    for row_idx, f in enumerate(X_arr):
        has_hand2 = np.any(f[66:126] != 0)
        h1 = f[:63].copy()
        for i in range(0, 63, 3):
            h1[i] = -h1[i]
        if has_hand2:
            h2 = f[63:126].copy()
            for i in range(0, 63, 3):
                h2[i] = -h2[i]
            X_flip[row_idx, :63] = h2
            X_flip[row_idx, 63:126] = h1
            X_flip[row_idx, 126] = -f[126]
            X_flip[row_idx, 127] = f[127]
            X_flip[row_idx, 128] = f[128]
        else:
            X_flip[row_idx, :63] = h1
    return X_flip


def careful_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_idx: list[int] = []
    test_idx: list[int] = []
    for sign in sorted(set(labels)):
        group_idx = [i for i, value in enumerate(labels) if value == sign]
        cut = int(len(group_idx) * 0.7)
        if cut == 0 or cut == len(group_idx):
            raise ValueError(f"Sign {sign!r} does not have enough rows for a split")
        train_idx.extend(group_idx[:cut])
        test_idx.extend(group_idx[cut:])
    return np.array(train_idx), np.array(test_idx)


def make_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=10,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "knn": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    predictions = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }


def main() -> None:
    default_csv = project_root() / "models" / "sign_classifier" / "models" / "landmarks.csv"
    if not default_csv.exists():
        default_csv = project_root() / "models" / "sign_classifier" / "results" / "landmarks.csv"
    if not default_csv.exists():
        default_csv = project_root() / "results" / "landmarks.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=default_csv,
        help="Landmark CSV path",
    )
    args = parser.parse_args()

    x, y = load_landmarks(args.data)
    train_idx, test_idx = careful_split(y)
    random_train, random_test = train_test_split(
        np.arange(len(y)),
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Rows: {len(y)} | Features: {x.shape[1]} | Signs: {len(set(y))}")
    print("Pictures per sign:", dict(sorted(Counter(y).items())))
    print("\nCareful per-sign split (first 70% train, last 30% test):")

    models_dir = project_root() / "models" / "sign_classifier" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    careful_scores: dict[str, dict[str, float]] = {}

    # Prepare augmented training data (original + horizontal flip)
    x_train_aug = np.vstack([x[train_idx], flip_features(x[train_idx])])
    y_train_aug = np.concatenate([y[train_idx], y[train_idx]])

    x_rand_train_aug = np.vstack([x[random_train], flip_features(x[random_train])])
    y_rand_train_aug = np.concatenate([y[random_train], y[random_train]])

    for name, model in make_models().items():
        model.fit(x_train_aug, y_train_aug)
        scores = evaluate(model, x[test_idx], y[test_idx])
        careful_scores[name] = scores
        rows.append({"model": name, "split": "careful", **scores})
        print(
            f"{name:24} accuracy={scores['accuracy']:.3f} "
            f"balanced_accuracy={scores['balanced_accuracy']:.3f} "
            f"macro_f1={scores['macro_f1']:.3f}"
        )

        model.fit(x_rand_train_aug, y_rand_train_aug)
        random_scores = evaluate(model, x[random_test], y[random_test])
        rows.append({"model": name, "split": "random", **random_scores})
        print(
            f"{'':24} random accuracy={random_scores['accuracy']:.3f} "
            f"balanced_accuracy={random_scores['balanced_accuracy']:.3f} "
            f"macro_f1={random_scores['macro_f1']:.3f}"
        )

    best_name = max(
        careful_scores,
        key=lambda name: (
            careful_scores[name]["macro_f1"],
            careful_scores[name]["balanced_accuracy"],
            careful_scores[name]["accuracy"],
        ),
    )

    print(f"\nBest careful-split model: {best_name}")
    print(f"Training final {best_name} model on all rows with flip augmentation...")
    x_all_aug = np.vstack([x, flip_features(x)])
    y_all_aug = np.concatenate([y, y])
    final_model = make_models()[best_name]
    final_model.fit(x_all_aug, y_all_aug)

    # Save only the best model as model.joblib
    joblib.dump(final_model, models_dir / "model.joblib")
    print(f"Saved best model ({best_name}) as models/sign_classifier/models/model.joblib")

    metrics_path = models_dir / "comparison_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "split", "accuracy", "balanced_accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics saved in {metrics_path}")


if __name__ == "__main__":
    main()

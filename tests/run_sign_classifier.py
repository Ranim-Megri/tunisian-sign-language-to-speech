"""run_sign_classifier.py: Test script for the Tunisian Sign Language Classifier.

Uses the best-performing model (rbf_svm) saved as model.joblib
in models/sign_classifier/models/.

Usage:
    # 1. Real-time webcam sign recognition (requires camera):
    python tests/run_sign_classifier.py --live

    # 2. Classify a single sign image:
    python tests/run_sign_classifier.py --image "path/to/sign.jpg"

    # 3. Simulate classification with sign words (no camera needed):
    python tests/run_sign_classifier.py --test-signs sbitar taxi

    # 4. Show model info:
    python tests/run_sign_classifier.py --info
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_models_dir = PROJECT_ROOT / "models"
_training_dir = PROJECT_ROOT / "models" / "sign_classifier" / "training"
_inference_dir = PROJECT_ROOT / "models" / "sign_classifier" / "inference"
for p in [str(PROJECT_ROOT), str(_models_dir), str(_training_dir), str(_inference_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

CLASSIFIER_MODELS_DIR = PROJECT_ROOT / "models" / "sign_classifier" / "models"


def cmd_info():
    """Print info about the installed classifier model."""
    import joblib
    model_path = CLASSIFIER_MODELS_DIR / "model.joblib"
    if not model_path.exists():
        print(f"[ERROR] No model found at {model_path}")
        return

    model = joblib.load(model_path)
    classes = list(model.classes_)
    print("=" * 60)
    print("SIGN CLASSIFIER MODEL INFO")
    print("=" * 60)
    print(f"  Model file : {model_path}")
    print(f"  Model type : {type(model).__name__}")
    print(f"  Classes ({len(classes)}): {', '.join(sorted(classes))}")

    metrics_csv = CLASSIFIER_MODELS_DIR / "comparison_metrics.csv"
    if metrics_csv.exists():
        import csv
        with open(metrics_csv, "r", encoding="utf-8") as f:
            best_row = None
            for row in csv.DictReader(f):
                if row["model"] == "rbf_svm" and row["split"] == "careful":
                    best_row = row
            if best_row:
                print(f"\n  Performance (careful split):")
                print(f"    Accuracy           : {float(best_row['accuracy']):.3f}")
                print(f"    Balanced accuracy  : {float(best_row['balanced_accuracy']):.3f}")
                print(f"    Macro F1           : {float(best_row['macro_f1']):.3f}")
    print("=" * 60)


def cmd_live(camera_id: int, confidence: float):
    """Launch the live webcam classifier."""
    from test_live import run_live
    run_live(camera_id=camera_id, min_confidence=confidence, model_name="model.joblib")


def cmd_image(image_path: str, confidence: float):
    """Classify a single image file end-to-end."""
    import cv2
    import joblib
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from features import landmarks_to_features

    model_path = CLASSIFIER_MODELS_DIR / "model.joblib"
    if not model_path.exists():
        print(f"[ERROR] No model found at {model_path}")
        return

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        return

    task_path = PROJECT_ROOT / "assets" / "sign_data" / "hand_landmarker.task"
    if not task_path.exists():
        task_path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    if not task_path.exists():
        print(f"[ERROR] MediaPipe hand landmarker task not found.")
        return

    model = joblib.load(model_path)
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )

    with vision.HandLandmarker.create_from_options(options) as hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hands.detect(mp_image)

        if not result.hand_landmarks:
            print("No hands detected in the image.")
            return

        feats = landmarks_to_features(result.hand_landmarks)
        if feats is None:
            print("Could not extract features from detected hands.")
            return

        probs = model.predict_proba([feats])[0]
        best_idx = int(np.argmax(probs))
        sign = model.classes_[best_idx]
        conf = float(probs[best_idx])

        if conf >= confidence:
            print(f"✓ Prediction: [{sign}] with {conf * 100:.1f}% confidence")
        else:
            print(f"? Low confidence: [{sign}] with {conf * 100:.1f}% (threshold: {confidence * 100:.0f}%)")


def cmd_test_signs(words: list):
    """Simulate pipeline with given sign words (no camera needed)."""
    from tts.engine import TunisianTTSEngine

    print("=" * 60)
    print("SIGN-TO-SPEECH SIMULATION (no camera)")
    print(f"Simulating recognized signs: {words}")
    print("=" * 60)

    engine = TunisianTTSEngine()
    out_wav, audio, fs = engine.synthesize(words)
    dur = len(audio) / fs
    print(f"✓ Output: {out_wav} ({dur:.2f}s, {fs} Hz)")


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Tunisian Sign Language Classifier – test runner."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true",
                       help="Launch live webcam sign classifier")
    group.add_argument("--image", type=str, default=None,
                       help="Classify a single sign image")
    group.add_argument("--test-signs", nargs="+", default=None,
                       help="Simulate pipeline with sign words (no camera)")
    group.add_argument("--info", action="store_true",
                       help="Show installed model info and metrics")

    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for live mode (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.60,
                        help="Confidence threshold (default: 0.60)")
    args = parser.parse_args()

    if args.info:
        cmd_info()
    elif args.image:
        cmd_image(args.image, args.confidence)
    elif args.test_signs:
        cmd_test_signs(args.test_signs)
    else:
        # Default to --live
        cmd_live(args.camera, args.confidence)


if __name__ == "__main__":
    main()

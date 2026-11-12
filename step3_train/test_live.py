"""live_test.py: show the model's guess live from the webcam. Press q to quit.
Run:  python step3_train/live_test.py"""
from pathlib import Path
import argparse
import urllib.request
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from features import get_features
"""python step3_train\test_live.py --model rbf_svm.joblib
python step3_train\test_live.py --model extra_trees.joblib
python step3_train\test_live.py --model random_forest.joblib"""

MIN_CONFIDENCE = 0.6      # below this we show "?" instead of a guess

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MODEL_PATH = PROJECT / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print("Downloading the MediaPipe hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

parser = argparse.ArgumentParser(description="Test a trained sign model with the webcam.")
parser.add_argument(
    "--model",
    default="model.joblib",
    help="Model filename in models/ or a model path (default: model.joblib)",
)
args = parser.parse_args()
model_path = Path(args.model)
if not model_path.is_absolute():
    model_path = (PROJECT / "models" / model_path) if model_path.parent == Path(".") else PROJECT / model_path
if not model_path.exists() and args.model == "model.joblib":
    model_path = HERE / "model.joblib"
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")
model = joblib.load(model_path)
print(f"Using model: {model_path}")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open the webcam.")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
)
with vision.HandLandmarker.create_from_options(options) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feats = get_features(hands, frame)
        if feats is None:
            text = "no hand"
        else:
            probs = model.predict_proba([feats])[0]
            best = probs.argmax()
            text = f"{model.classes_[best]}  {probs[best]:.2f}" if probs[best] >= MIN_CONFIDENCE else "?"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("TunSL test (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
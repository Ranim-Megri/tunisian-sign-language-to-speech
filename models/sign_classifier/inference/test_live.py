"""live_test.py: show the model's guess live from the webcam. Press q to quit."""
import sys
import time
from pathlib import Path
import argparse
import urllib.request
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT = Path(__file__).resolve().parents[3]
training_dir = PROJECT / "models" / "sign_classifier" / "training"
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from features import get_features, landmarks_to_features

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # wrist-pinky
)

MIN_CONFIDENCE = 0.6  # below this we show "?" instead of a guess

MODEL_PATH = PROJECT / "assets" / "sign_data" / "hand_landmarker.task"
if not MODEL_PATH.exists():
    alt_model = PROJECT / "models" / "hand_landmarker.task"
    if alt_model.exists():
        MODEL_PATH = alt_model

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the MediaPipe hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _open_camera(camera_id: int):
    backends = [cv2.CAP_ANY]
    if sys.platform == "win32":
        backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])

    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                for _ in range(8):
                    ok, test_frame = cap.read()
                    if ok and test_frame is not None and test_frame.size > 0:
                        backend_name = "DEFAULT" if backend == cv2.CAP_ANY else ("DSHOW" if backend == cv2.CAP_DSHOW else "MSMF")
                        print(f"Connected to camera {camera_id} via {backend_name} ({test_frame.shape[1]}x{test_frame.shape[0]}).")
                        return cap
                    time.sleep(0.05)
            cap.release()
        except Exception:
            pass

    for alt_id in [0, 1, 2, 3]:
        if alt_id != camera_id:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(alt_id, backend)
                    if cap.isOpened():
                        for _ in range(8):
                            ok, test_frame = cap.read()
                            if ok and test_frame is not None and test_frame.size > 0:
                                print(f"Webcam found and verified on alternative index {alt_id} ({test_frame.shape[1]}x{test_frame.shape[0]})!")
                                return cap
                            time.sleep(0.05)
                    cap.release()
                except Exception:
                    pass
    return None


def run_live(camera_id: int = 0, min_confidence: float = MIN_CONFIDENCE, model_name: str = "model.joblib"):
    model_path = Path(model_name)
    if not model_path.is_absolute():
        models_dir = PROJECT / "models" / "sign_classifier" / "models"
        model_path = (models_dir / model_path) if model_path.parent == Path(".") else PROJECT / model_path
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    print(f"Using model: {model_path}")
    cap = _open_camera(camera_id)
    if cap is None:
        print("\n" + "=" * 60)
        print(f"[ERROR] Cannot open webcam with index {camera_id}.")
        print("1. Check that a webcam is plugged in and recognized.")
        print("2. Ensure no other application (Zoom, Teams, Chrome) is using it.")
        print("3. Check Windows Privacy: Settings > Privacy & security > Camera > 'Let desktop apps access your camera'.")
        print("4. Try another index: python test_live.py --camera 1 (or 2).")
        print("=" * 60)
        return

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )

    hand_colors = [
        {"bone": (0, 215, 255), "joint": (0, 255, 0)},     # Hand 1: Yellow bones, Green joints
        {"bone": (255, 200, 50), "joint": (255, 80, 220)}   # Hand 2: Cyan bones, Magenta joints
    ]
    mirror = True

    win_name = "TunSL test (press q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 640)
    print("Starting test_live. Press [Q] to quit.")

    failed_reads = 0
    try:
        with vision.HandLandmarker.create_from_options(options) as hands:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    failed_reads += 1
                    if failed_reads <= 15:
                        time.sleep(0.04)
                        continue
                    else:
                        print(f"\n[ERROR] Camera device {camera_id} stopped delivering video frames.")
                        break
                failed_reads = 0

                if mirror:
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hands.detect(mp_image)

                n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
                if result.hand_landmarks:
                    feats = landmarks_to_features(result.hand_landmarks)
                    if feats is not None:
                        probs = model.predict_proba([feats])[0]
                        best = probs.argmax()
                        text = f"{model.classes_[best]}  {probs[best]:.2f}" if probs[best] >= min_confidence else f"? ({model.classes_[best]} {probs[best]:.2f})"
                    else:
                        text = "no hand"

                    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                        colors = hand_colors[min(hand_idx, len(hand_colors) - 1)]
                        pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]
                        for p1, p2 in HAND_CONNECTIONS:
                            if p1 < len(pts) and p2 < len(pts):
                                cv2.line(frame, pts[p1], pts[p2], colors["bone"], 2)
                        for pt in pts:
                            cv2.circle(frame, pt, 5, colors["joint"], -1)
                        if pts:
                            cv2.putText(frame, f"H{hand_idx + 1}", (pts[0][0] - 10, pts[0][1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["joint"], 2)
                else:
                    text = "no hand"

                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                hands_color = (0, 255, 120) if n_hands == 2 else ((0, 215, 255) if n_hands == 1 else (120, 120, 120))
                cv2.putText(frame, f"[Hands: {n_hands}/2]", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hands_color, 2)
                cv2.putText(frame, "[M] Mirror  [Q] Quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow(win_name, frame)
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord("m"), ord("M")):
                    mirror = not mirror
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Exception in video loop: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")


def main():
    parser = argparse.ArgumentParser(description="Test a trained sign model with the webcam.")
    parser.add_argument(
        "--model",
        default="model.joblib",
        help="Model filename in models/sign_classifier/models/ or a model path (default: model.joblib)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--confidence", type=float, default=MIN_CONFIDENCE, help="Minimum confidence threshold")
    args = parser.parse_args()
    run_live(camera_id=args.camera, min_confidence=args.confidence, model_name=args.model)


if __name__ == "__main__":
    main()

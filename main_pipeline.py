import argparse
import sys
import time
from collections import deque, Counter
from pathlib import Path
from typing import List, Optional, Union

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
training_dir = PROJECT_ROOT / "models" / "sign_classifier" / "training"
if not training_dir.exists():
    training_dir = PROJECT_ROOT / "sign_classifier" / "training"
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))
models_dir = PROJECT_ROOT / "models"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from features import get_features, landmarks_to_features
from tts.engine import TunisianTTSEngine

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # wrist-pinky
)

TASK_PATH = PROJECT_ROOT / "assets" / "sign_data" / "hand_landmarker.task"
if not TASK_PATH.exists():
    for fallback in [
        PROJECT_ROOT / "models" / "sign_classifier" / "models" / "hand_landmarker.task",
        PROJECT_ROOT / "models" / "hand_landmarker.task",
        PROJECT_ROOT / "assets" / "audio" / "hand_landmarker.task",
    ]:
        if fallback.exists():
            TASK_PATH = fallback
            break

MODELS_DIR = PROJECT_ROOT / "models" / "sign_classifier" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = PROJECT_ROOT / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = PROJECT_ROOT / "sign_classifier" / "models"


class SignToSpeechPipeline:
    def __init__(
        self,
        model_name: str = "model.joblib",
        camera_id: int = 0,
        min_confidence: float = 0.65,
        commit_threshold: int = 4,
        pause_timeout: float = 1.2,
        mirror: bool = True,
        auto_speak: bool = True
    ):
        self.camera_id = camera_id
        self.min_confidence = min_confidence
        self.commit_threshold = commit_threshold
        self.pause_timeout = pause_timeout
        self.mirror = mirror
        self.auto_speak = auto_speak

        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Sign classifier model not found: {model_path}")

        print(f"Loading best performing classifier: {model_path.name}")
        self.model = joblib.load(model_path)
        self.classes = list(self.model.classes_)

        print("Initializing Tunisian Derja TTS Engine...")
        self.tts = TunisianTTSEngine()

        if not TASK_PATH.exists():
            raise FileNotFoundError(f"MediaPipe task not found: {TASK_PATH}")

        self.options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(TASK_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35
        )

        self.vote_buffer = deque(maxlen=8)
        self.sentence_words: List[str] = []
        self.last_sign_time = time.time()
        self.last_speech_time = 0.0
        self.last_spoken_sign: Optional[str] = None
        self.status_message = "Ready. Show hands to begin signing."
        self.is_speaking = False
        self.detected_hands_count = 0

    def _open_camera(self, camera_id: int):
        backends = [cv2.CAP_ANY]
        if sys.platform == "win32":
            backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])

        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    # Test that frames can actually be captured from this capture device
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
                                    self.camera_id = alt_id
                                    return cap
                                time.sleep(0.05)
                        cap.release()
                    except Exception:
                        pass
        return None

    def process_frame(self, landmarker, frame: np.ndarray):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        current_sign = None
        current_conf = 0.0
        self.detected_hands_count = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0

        # Colors for Hand 1 and Hand 2
        hand_colors = [
            {"bone": (0, 215, 255), "joint": (0, 255, 0)},     # Hand 1: Yellow bones, Green joints
            {"bone": (255, 200, 50), "joint": (255, 80, 220)}   # Hand 2: Cyan bones, Magenta joints
        ]

        if detection_result.hand_landmarks:
            feats = landmarks_to_features(detection_result.hand_landmarks)
            if feats is not None:
                probs = self.model.predict_proba([feats])[0]
                best_idx = int(np.argmax(probs))
                current_conf = float(probs[best_idx])
                if current_conf >= self.min_confidence:
                    current_sign = self.classes[best_idx]

            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                colors = hand_colors[min(hand_idx, len(hand_colors) - 1)]
                pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]
                for p1, p2 in HAND_CONNECTIONS:
                    if p1 < len(pts) and p2 < len(pts):
                        cv2.line(frame, pts[p1], pts[p2], colors["bone"], 2)
                for pt in pts:
                    cv2.circle(frame, pt, 5, colors["joint"], -1)

                # Label hand index on wrist
                if pts:
                    label = f"H{hand_idx + 1}"
                    cv2.putText(frame, label, (pts[0][0] - 10, pts[0][1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["joint"], 2)

        if current_sign:
            self.vote_buffer.append(current_sign)
            self.last_sign_time = time.time()
            counts = Counter([s for s in self.vote_buffer if s is not None])
            if counts:
                most_common_sign, count = counts.most_common(1)[0]
                if most_common_sign is not None and count >= self.commit_threshold:
                    most_common_sign = str(most_common_sign)
                    now = time.time()
                    sign_is_new = not self.sentence_words or self.sentence_words[-1] != most_common_sign
                    if sign_is_new:
                        self.sentence_words.append(most_common_sign)

                    # AUTOMATIC SPEECH: Vocalize immediately upon valid sign recognition!
                    if self.auto_speak and (most_common_sign != self.last_spoken_sign or (now - self.last_speech_time) > 2.0):
                        self.last_spoken_sign = most_common_sign
                        self.trigger_speech([most_common_sign])

                    self.vote_buffer.clear()
                    self.status_message = f"Detected: {most_common_sign}"
        else:
            if self.vote_buffer:
                self.vote_buffer.popleft()
            # If multiple signs chained and user pauses, vocalize composite sentence
            valid_words = [str(w) for w in self.sentence_words if w is not None and str(w).strip()]
            if len(valid_words) > 1 and (time.time() - self.last_sign_time) > self.pause_timeout:
                self.sentence_words.clear()
                self.last_spoken_sign = None
                self.trigger_speech(valid_words)

        self._draw_hud(frame, current_sign, current_conf)
        return frame, current_sign, current_conf

    def trigger_speech(self, words: Optional[List[str]] = None):
        raw_words = list(words) if words else list(self.sentence_words)
        words_to_speak = [str(w) for w in raw_words if w is not None and str(w).strip()]
        if not words_to_speak or self.is_speaking:
            return

        self.is_speaking = True
        self.status_message = f"Speaking: {' '.join(words_to_speak)}"
        print(f"\n>>> [AUTO-SPEECH TRIGGERED] Playing Tunisian voice for: {words_to_speak}")

        try:
            out_wav, _, _ = self.tts.synthesize(words_to_speak)
            self.tts.player.play_wav(out_wav, async_play=True)
            self.last_speech_time = time.time()
        except Exception as e:
            print(f"[TTS Error] {e}")
        finally:
            self.is_speaking = False

    def _draw_hud(self, frame: np.ndarray, sign: Optional[str], conf: float):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if sign:
            pred_text = f"Sign: {sign} ({int(conf * 100)}%)"
            color = (0, 255, 120)
        else:
            pred_text = "Sign: Waiting for stable pose..."
            color = (180, 180, 180)
        cv2.putText(frame, pred_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        # Hands count badge
        n_hands = self.detected_hands_count
        if n_hands == 2:
            hands_color = (0, 255, 120)
            hands_text = "[Hands: 2/2 OK]"
        elif n_hands == 1:
            hands_color = (0, 215, 255)
            hands_text = "[Hands: 1/2]"
        else:
            hands_color = (120, 120, 120)
            hands_text = "[Hands: 0/2]"
        cv2.putText(frame, hands_text, (w - 530, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, hands_color, 2)

        speech_mode_text = "VOICE: AUTO" if self.auto_speak else "VOICE: MANUAL"
        speech_color = (0, 255, 120) if self.auto_speak else (150, 150, 150)
        cv2.putText(frame, f"[{speech_mode_text}]", (w - 365, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, speech_color, 2)

        mirror_text = "MIRROR: ON" if self.mirror else "MIRROR: OFF"
        cv2.putText(frame, f"[{mirror_text}]", (w - 200, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 255), 2)

        valid_sentence = [str(x) for x in self.sentence_words if x is not None and str(x).strip()]
        sentence_str = " + ".join([f"[{x}]" for x in valid_sentence])
        if not sentence_str:
            sentence_str = "None (sign triggers speech automatically)"
        cv2.putText(frame, f"Sentence: {sentence_str}", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {self.status_message}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        help_text = "[S] Toggle Auto-Voice   [M] Mirror   [SPACE] Speak   [C] Clear   [Q] Quit"
        cv2.putText(frame, help_text, (w - 610, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    def run_live(self):
        print(f"Opening webcam device {self.camera_id}...")
        cap = self._open_camera(self.camera_id)
        if cap is None:
            print("\n" + "=" * 60)
            print(f"[ERROR] Unable to open camera device (index {self.camera_id}).")
            print("=" * 60)
            print("Common causes and solutions:")
            print("1. No camera hardware: Ensure a physical webcam or built-in camera is connected and recognized in Windows Device Manager.")
            print("2. Camera is in use: Close any application that might be using the camera (Zoom, Teams, Discord, Chrome, or Windows Camera App).")
            print("3. Windows Privacy Settings: Check Start > Settings > Privacy & security > Camera:")
            print("   - Ensure 'Camera access' is ON.")
            print("   - Ensure 'Let desktop apps access your camera' is ON.")
            print("4. Camera Index: If your camera is on a different port/index, specify it:")
            print("   python main_pipeline.py --camera 1   (or --camera 2)")
            print("\nAlternative test modes available right now:")
            print("  - Test without webcam (simulated signs):")
            print("      python main_pipeline.py --test-signs sbitar taxi")
            print("  - Test on a sign photo:")
            print("      python main_pipeline.py --image \"assets/sign_data/(First ever) Tunisian Sign Language Dataset/Data/Destinations/sbitar/sbitar (1).jpg\"")
            print("=" * 60)
            return

        win_name = "Tunisian Sign Language to Speech Pipeline"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 960, 640)
        print("Starting live pipeline. Press [Q] to exit.")

        failed_reads = 0
        try:
            with vision.HandLandmarker.create_from_options(self.options) as landmarker:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        failed_reads += 1
                        if failed_reads <= 15:
                            time.sleep(0.04)
                            continue
                        else:
                            print(f"\n[ERROR] Camera device {self.camera_id} stopped delivering video frames.")
                            break
                    failed_reads = 0

                    if self.mirror:
                        frame = cv2.flip(frame, 1)

                    self.process_frame(landmarker, frame)
                    cv2.imshow(win_name, frame)

                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord(" "):
                        self.trigger_speech()
                    elif key in (ord("s"), ord("S")):
                        self.auto_speak = not self.auto_speak
                        self.status_message = f"Auto-Speech {'enabled' if self.auto_speak else 'disabled'}."
                    elif key in (ord("m"), ord("M")):
                        self.mirror = not self.mirror
                        self.status_message = f"Mirror mode {'enabled' if self.mirror else 'disabled'}."
                    elif key in (ord("c"), ord("C")):
                        self.sentence_words.clear()
                        self.vote_buffer.clear()
                        self.status_message = "Cleared sentence buffer."
        except KeyboardInterrupt:
            print("\nPipeline stopped by user.")
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Exception in video loop: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam released and window closed.")

    def run_image(self, image_path: Union[str, Path]):
        image_path = Path(image_path)
        print("=" * 60)
        print(f"TUNISIAN SIGN-TO-SPEECH IMAGE INFERENCE: {image_path.name}")
        print("=" * 60)
        if not image_path.exists():
            print(f"Error: Image not found at {image_path}")
            return

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image at {image_path}")
            return

        with vision.HandLandmarker.create_from_options(self.options) as landmarker:
            _, sign, conf = self.process_frame(landmarker, frame)
            if sign:
                print(f"Prediction: [{sign}] with {conf * 100:.1f}% confidence")
                print(f"Triggering Tunisian TTS for '{sign}'...")
                self.sentence_words = [sign]
                self.trigger_speech()
            else:
                print("No hand landmark detected with sufficient confidence in this image.")

    def run_simulation(self, test_words: List[str]):
        print("=" * 60)
        print("TUNISIAN SIGN-TO-SPEECH PIPELINE SIMULATION")
        print(f"Simulating recognized signs: {test_words}")
        print("=" * 60)
        out_wav, audio, fs = self.tts.synthesize(test_words)
        dur = len(audio) / fs
        print(f"Output Audio: {out_wav} ({dur:.2f}s, {fs}Hz)")
        print("Playing voice output through PC speakers...")
        self.tts.player.play_wav(out_wav)
        print("Simulation completed successfully.")


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="End-to-end Tunisian Sign Language to Speech PC Pipeline.")
    parser.add_argument("--model", type=str, default="model.joblib", help="Classifier model in sign_classifier/models/ (default: model.joblib)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--confidence", type=float, default=0.55, help="Minimum sign prediction confidence threshold (default: 0.55)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror mode for webcam")
    parser.add_argument("--image", type=str, default=None, help="Process a single sign image end-to-end")
    parser.add_argument("--test-signs", nargs="+", default=None, help="Run direct simulation with sign words (e.g. sbitar taxi)")
    args = parser.parse_args()

    pipeline = SignToSpeechPipeline(
        model_name=args.model,
        camera_id=args.camera,
        min_confidence=args.confidence,
        mirror=not args.no_mirror
    )

    if args.image:
        pipeline.run_image(args.image)
    elif args.test_signs:
        pipeline.run_simulation(args.test_signs)
    else:
        pipeline.run_live()


if __name__ == "__main__":
    main()

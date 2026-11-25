"""features.py: turns one picture (a camera frame) into 129 numbers.
This must be the SAME calculation as get_features in extract_landmarks.py."""
import cv2
import numpy as np
import mediapipe as mp


def landmarks_to_features(hand_landmarks_list):
    """Calculate 129 features directly from detected hand landmarks list."""
    if not hand_landmarks_list:
        return None

    found_hands = []
    for hand in hand_landmarks_list:
        pts = np.array([[p.x, p.y, p.z] for p in hand])
        wrist = pts[0].copy()
        pts = pts - wrist
        size = np.linalg.norm(pts[9]) + 1e-6
        found_hands.append((wrist, (pts / size).flatten()))
    found_hands.sort(key=lambda h: h[0][0])

    feats = np.zeros(129)
    feats[:63] = found_hands[0][1]
    if len(found_hands) >= 2:
        feats[63:126] = found_hands[1][1]
        feats[126:129] = found_hands[1][0] - found_hands[0][0]
    return feats


def get_features(landmarker, img):
    """Return 129 numbers for one picture (BGR image), or None if no hand is found."""
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(image)
    return landmarks_to_features(result.hand_landmarks)
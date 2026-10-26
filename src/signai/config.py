"""Central configuration: paths, feature geometry, and the demo vocabulary."""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # src/signai
PROJECT_ROOT = PACKAGE_DIR.parent.parent               # repository root

DATA_DIR = PROJECT_ROOT / "data"
SEQUENCES_DIR = DATA_DIR / "sequences"                 # dynamic samples: (T, 126) per file
STATIC_DIR = DATA_DIR / "static"                       # static samples: (126,) per file
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

# ------------------------------------------------------------------ geometry --
NUM_LANDMARKS = 21
NUM_COORDS = 3                                         # x, y, z from MediaPipe
FEATS_PER_HAND = NUM_LANDMARKS * NUM_COORDS            # 63
MAX_HANDS = 2
FRAME_FEATURES = FEATS_PER_HAND * MAX_HANDS            # 126 (hand A | hand B)

DEFAULT_SEQ_LEN = 30                                   # frames per dynamic sample (~1 s @ 30 fps)

# ------------------------------------------------------- MediaPipe model files --
GESTURE_MODEL_FILENAME = "gesture_recognizer.task"
GESTURE_MODEL_PATH = MODELS_DIR / GESTURE_MODEL_FILENAME
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# Landmarks-only fallback backend (used when the gesture model is unavailable).
HAND_LANDMARK_FILENAME = "hand_landmarker.task"
HAND_LANDMARK_PATH = MODELS_DIR / HAND_LANDMARK_FILENAME
HAND_LANDMARK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ----------------------------------------------------------- demo vocabulary --
# Mapping from pretrained hand-pose categories to display words.
#
# IMPORTANT: this is a gesture -> word DEMO mapping so the app works out of the
# box. It is NOT a faithful ASL dictionary. Real ASL words such as "thank you"
# or "hello" depend on movement and context and are covered by the custom
# training pipeline (collect_data.py + train.py).
DEMO_WORD_MAP: dict[str, str] = {
    "Open_Palm": "Hello",
    "ILoveYou": "I Love You",
    "Victory": "Peace",
    "Thumb_Up": "Good",
    "Thumb_Down": "Bad",
    "Pointing_Up": "You",
    "Closed_Fist": "Stop",
    "None": "",
}

# Labels the classifier can emit (excluding "None").
GESTURE_LABELS: list[str] = [k for k in DEMO_WORD_MAP if k != "None"]


def word_for_gesture(label: str) -> str:
    """Translate a raw gesture label into its demo display word."""
    return DEMO_WORD_MAP.get(label, "")

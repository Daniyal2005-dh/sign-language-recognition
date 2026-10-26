"""MediaPipe hand-landmark pipeline with interchangeable backends.

Backends (first available wins):
1. **Gesture Recognizer** (``gesture_recognizer.task``): 21 landmarks *and*
   pretrained gesture categories in a single call.
2. **Hand Landmarker** (``hand_landmarker.task``): landmarks only; gesture
   labels come from :mod:`signai.gestures` rule-based classification.
   (MediaPipe 1.x removed the legacy ``mp.solutions`` API, so the tasks API is
   used for both backends.)

Both backends return the same uniform :class:`HandFrame`, so callers never
need to know which one is running. Model files are downloaded on first use
(and are normally committed to the repo for offline/cloud use).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from .config import (
    GESTURE_MODEL_PATH,
    GESTURE_MODEL_URL,
    HAND_LANDMARK_PATH,
    HAND_LANDMARK_MODEL_URL,
)
from .gestures import NONE_LABEL, classify_static

logger = logging.getLogger(__name__)

# 21-landmark skeleton edges used for drawing overlays.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
    (9, 10), (10, 11), (11, 12), (9, 13),
    (13, 14), (14, 15), (15, 16), (13, 17),
    (17, 18), (18, 19), (19, 20), (0, 17),
)


@dataclass
class HandData:
    """One detected hand."""

    landmarks: np.ndarray          # (21, 3), normalised image coordinates
    gesture: str = NONE_LABEL      # gesture category (or heuristic label)
    score: float = 0.0             # confidence in [0, 1]
    handedness: str = ""           # informational only; features never rely on it


@dataclass
class HandFrame:
    """All hands detected in one video frame."""

    hands: list[HandData] = field(default_factory=list)
    backend: str = ""

    def __bool__(self) -> bool:
        return len(self.hands) > 0

    def best_hand(self) -> HandData | None:
        """The hand with the most confident gesture (for one-handed signing)."""
        if not self.hands:
            return None
        return max(self.hands, key=lambda h: h.score)


def _download_model(path, url: str, timeout: int = 120) -> bool:
    """Download a ``.task`` model file unless a valid copy already exists."""
    try:
        if path.exists() and path.stat().st_size > 1_000_000:
            return True
    except OSError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        import urllib.request

        logger.info("Downloading MediaPipe model %s ...", path.name)
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(path)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Could not download %s (%s)", path.name, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def ensure_gesture_model(path=GESTURE_MODEL_PATH, url: str = GESTURE_MODEL_URL, **kw) -> bool:
    return _download_model(path, url, **kw)


def ensure_hand_landmark_model(path=HAND_LANDMARK_PATH, url: str = HAND_LANDMARK_MODEL_URL, **kw) -> bool:
    return _download_model(path, url, **kw)


class HandPipeline:
    """Detect hands in BGR frames; thread-safe (one lock around inference)."""

    def __init__(self, num_hands: int = 2, use_task_model: bool = True):
        self.num_hands = num_hands
        self.backend = ""
        self._frame_index = 0
        self._lock = threading.Lock()
        self._recognizer = None
        self._landmarker = None
        self._init(use_task_model)

    # ------------------------------------------------------------------ setup
    def _init(self, use_task_model: bool) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        if use_task_model and ensure_gesture_model():
            try:
                options = vision.GestureRecognizerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(GESTURE_MODEL_PATH)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.num_hands,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._recognizer = vision.GestureRecognizer.create_from_options(options)
                self.backend = "mediapipe:gesture-recognizer"
                return
            except Exception as exc:  # pragma: no cover - version dependent
                logger.warning("GestureRecognizer unavailable (%s); trying HandLandmarker", exc)

        if ensure_hand_landmark_model():
            try:
                options = vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARK_PATH)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.num_hands,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
                self.backend = "mediapipe:hand-landmarker+rules"
                return
            except Exception as exc:  # pragma: no cover - version dependent
                logger.warning("HandLandmarker unavailable (%s)", exc)

        raise RuntimeError(
            "No MediaPipe model could be loaded. Run `python scripts/download_model.py` "
            "and check your internet connection."
        )

    # ----------------------------------------------------------------- infer
    def process_bgr(self, bgr: np.ndarray) -> HandFrame:
        """Run detection on one BGR image and return a uniform result."""
        with self._lock:
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            if self._recognizer is not None:
                frame = self._process_task(rgb, self._frame_index * 33)
            else:
                frame = self._process_landmarker(rgb, self._frame_index * 33)
            self._frame_index += 1
            return frame

    def _to_array(self, lm_list) -> np.ndarray:
        return np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)

    def _image(self, rgb: np.ndarray):
        return self._mp_image(rgb)

    def _mp_image(self, rgb: np.ndarray):
        import mediapipe as mp

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _process_task(self, rgb: np.ndarray, timestamp_ms: int) -> HandFrame:
        result = self._recognizer.recognize_for_video(self._mp_image(rgb), timestamp_ms)

        hands: list[HandData] = []
        gestures = result.gestures or []
        for idx, lm_list in enumerate(result.hand_landmarks or []):
            gesture, score = NONE_LABEL, 0.0
            if idx < len(gestures) and gestures[idx]:
                top = gestures[idx][0]
                gesture, score = top.category_name, float(top.score)
            handedness = ""
            if idx < len(result.handedness or []) and result.handedness[idx]:
                handedness = result.handedness[idx][0].category_name
            hands.append(HandData(self._to_array(lm_list), gesture, score, handedness))
        return finalize_frame(hands, self.backend)

    def _process_landmarker(self, rgb: np.ndarray, timestamp_ms: int) -> HandFrame:
        result = self._landmarker.detect_for_video(self._mp_image(rgb), timestamp_ms)

        hands: list[HandData] = []
        for idx, lm_list in enumerate(result.hand_landmarks or []):
            gesture, score = classify_static(self._to_array(lm_list))
            handedness = ""
            if result.handedness and idx < len(result.handedness):
                handedness = result.handedness[idx][0].category_name
            hands.append(HandData(self._to_array(lm_list), gesture, score, handedness))
        return finalize_frame(hands, self.backend)

    def close(self) -> None:
        with self._lock:
            if self._recognizer is not None:
                self._recognizer.close()
                self._recognizer = None
            if self._landmarker is not None:
                self._landmarker.close()
                self._landmarker = None


def finalize_frame(hands: list[HandData], backend: str) -> HandFrame:
    """Sort hands by wrist x for a stable slot order."""
    hands = sorted(hands, key=lambda h: float(h.landmarks[0][0]))
    return HandFrame(hands=hands, backend=backend)


# --------------------------------------------------------------------- drawing
def draw_hand_frame(
    bgr: np.ndarray,
    frame: HandFrame,
    text: str | None = None,
    conf: float | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Draw landmarks + status banner on the frame (for display)."""
    import cv2

    img = bgr
    h, w = img.shape[:2]

    for hand in frame.hands:
        pts = [(int(x * w), int(y * h)) for (x, y, _z) in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (170, 255, 120), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 4, (90, 200, 255), -1, cv2.LINE_AA)

    if text:
        label = text if conf is None else f"{text}  ({conf:.0%})"
        if fps is not None:
            label = f"{label}   |   {fps:.0f} FPS"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        origin = (12, h - 24)
        cv2.rectangle(img, (8, h - 24 - th - baseline), (20 + tw, h - 10), (30, 30, 30), -1)
        cv2.putText(img, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)

    return img

"""Landmark normalisation and per-frame feature construction (pure NumPy).

Each MediaPipe hand is 21 landmarks with (x, y, z) in normalised image
coordinates. Raw coordinates mix pose information with *where* the hand is in
the frame and how big it is, so before classification we:

1. translate the hand so the wrist is at the origin, and
2. divide by the palm size (wrist -> middle-finger MCP distance).

The result describes the hand's SHAPE, independent of position and scale.
A frame packs up to two hands into one flat vector:

    [ hand A (63) | hand B (63) ]  ->  126 features per frame

Hands are ordered by wrist x-position, which stays consistent while a person
signs, and the unused slot is simply zeros (ignored by the model via masking).
"""
from __future__ import annotations

import numpy as np

from .config import FEATS_PER_HAND, FRAME_FEATURES, NUM_LANDMARKS


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """Return a translation- and scale-invariant copy of one hand (21, 3)."""
    lm = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    lm = lm - lm[0]                        # wrist -> origin
    scale = float(np.linalg.norm(lm[9]))   # wrist -> middle-finger MCP
    if scale < 1e-6:
        scale = 1.0
    return (lm / scale).astype(np.float32)


def hand_features(landmarks: np.ndarray) -> np.ndarray:
    """Flatten one hand into a 63-dim vector."""
    return normalize_hand(landmarks).reshape(FEATS_PER_HAND)


def order_hands(hands) -> list:
    """Sort hands consistently (by wrist x ascending) so slot A/B is stable."""
    return sorted(hands, key=lambda lm: float(np.asarray(lm).reshape(NUM_LANDMARKS, 3)[0][0]))


def frame_features(hands) -> np.ndarray:
    """Build the (126,) feature vector for one video frame."""
    feats = np.zeros(FRAME_FEATURES, dtype=np.float32)
    for slot, lm in enumerate(order_hands(hands)[:2]):
        feats[slot * FEATS_PER_HAND : (slot + 1) * FEATS_PER_HAND] = hand_features(lm)
    return feats


def frame_has_hand(feats: np.ndarray) -> bool:
    """True when at least one hand slot is non-empty."""
    feats = np.asarray(feats, dtype=np.float32).reshape(-1)
    return bool(feats.any())

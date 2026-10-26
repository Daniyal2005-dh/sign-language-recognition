"""Rule-based fallback gesture classifier over 21 MediaPipe hand landmarks.

This is the *fallback* backend: it classifies hand poses from geometry alone,
using the SAME label vocabulary as MediaPipe's pretrained gesture recognizer
(``Open_Palm``, ``Victory``, ``Thumb_Up``, ...) so the rest of the pipeline is
backend-agnostic.

How it works
------------
For every finger we measure how straight the joint chain is, on a continuous
0..1 "extendedness" scale:

* each joint angle is computed between the landmark chain segments;
* a straight finger has ~180 degree angles, a curled one folds to well under
  140 degrees (the thumb uses both its MCP and IP angles because it can fold
  at the MCP while keeping the IP straight).

Class labels are simple rules over these certainties (e.g. "index and middle
extended, everything else curled" -> ``Victory``). Pure NumPy, no MediaPipe
import, so it is fully unit-testable in CI.
"""
from __future__ import annotations

import numpy as np

from .config import NUM_LANDMARKS
from .features import normalize_hand

NONE_LABEL = "None"

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Landmark index chains per finger: (base, pip, dip, tip). Thumb starts at CMC.
FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# Angle -> certainty mapping: 110 deg (folded) -> 0.0, 160+ deg (straight) -> 1.0.
_ANGLE_LO, _ANGLE_SPAN = 110.0, 50.0


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (degrees) at point ``b`` between segments b->a and b->c."""
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1)) * float(np.linalg.norm(v2))
    if n < 1e-9:
        return 0.0
    cosv = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def _chain_certainty(a_pip: float, a_dip: float) -> float:
    return _clip01(min((a_pip - _ANGLE_LO) / _ANGLE_SPAN, (a_dip - _ANGLE_LO) / _ANGLE_SPAN))


def finger_certainties(lm: np.ndarray) -> dict[str, float]:
    """Continuous certainty in [0, 1] that each finger is extended."""
    lm = np.asarray(lm, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    certs: dict[str, float] = {}

    for name, (b, pip, dip, tip) in FINGER_CHAINS.items():
        if name == "thumb":
            a_mcp = _angle_deg(lm[b], lm[pip], lm[dip])   # CMC - MCP - IP
            a_ip = _angle_deg(lm[pip], lm[dip], lm[tip])  # MCP - IP  - TIP
            certs[name] = _chain_certainty(a_mcp, a_ip)
        else:
            a_pip = _angle_deg(lm[b], lm[pip], lm[tip])   # MCP - PIP - TIP
            a_dip = _angle_deg(lm[pip], lm[dip], lm[tip]) # PIP - DIP - TIP
            certs[name] = _chain_certainty(a_pip, a_dip)

    return certs


def classify_static(landmarks: np.ndarray) -> tuple[str, float]:
    """Classify one hand's pose. Returns ``(label, confidence)``.

    Confidence is a heuristic certainty (mean of the finger certainties the
    rule relies on), not a model probability.
    """
    lm = normalize_hand(landmarks)
    c = finger_certainties(lm)

    is_ext = lambda f: c[f] >= 0.5  # noqa: E731
    mean_cert = lambda fs: float(np.mean([c[f] for f in fs])) if fs else 0.5  # noqa: E731

    t, i, m, r, p = (is_ext(f) for f in FINGERS)

    if t and i and m and r and p:
        return "Open_Palm", mean_cert(FINGERS)
    if i and m and not r and not p and not t:
        return "Victory", mean_cert(("index", "middle"))
    if t and i and p and not m and not r:
        return "ILoveYou", mean_cert(("thumb", "index", "pinky"))
    if i and not (m or r or p or t):
        return "Pointing_Up", mean_cert(("index",))
    if t and not (i or m or r or p):
        # y grows downward in image coordinates: thumb "up" = tip well above its MCP
        up = float(lm[4][1] - lm[2][1]) < -0.35
        return ("Thumb_Up" if up else "Thumb_Down"), mean_cert(("thumb",))
    if not (t or i or m or r or p):
        return "Closed_Fist", float(1.0 - np.mean([c[f] for f in FINGERS]))

    return NONE_LABEL, 0.0

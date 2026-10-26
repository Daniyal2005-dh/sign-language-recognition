"""Unit tests for the rule-based gesture classifier using synthetic poses."""
import numpy as np

from signai.gestures import NONE_LABEL, classify_static, finger_certainties

THUMB_POSES = {
    # collinear (straight) thumb pointing out to the left
    "straight": [(0.46, 0.47), (0.42, 0.44), (0.38, 0.41), (0.34, 0.38)],
    # folded across the palm (kinked at the MCP)
    "curled": [(0.44, 0.47), (0.41, 0.46), (0.435, 0.445), (0.46, 0.435)],
    # straight, pointing up
    "up": [(0.44, 0.47), (0.43, 0.42), (0.42, 0.37), (0.41, 0.32)],
    # straight, pointing down
    "down": [(0.44, 0.47), (0.43, 0.50), (0.42, 0.53), (0.41, 0.56)],
}


def _chain(base, direction=(0.0, -1.0), seg=0.035, straight=True):
    """4 landmark points for one finger chain (x, y)."""
    d = np.asarray(direction, dtype=np.float32)
    d = d / np.linalg.norm(d)
    base = np.asarray(base, dtype=np.float32)
    if straight:
        return [tuple(base + d * seg * k) for k in range(4)]
    # curled: pip slightly raised, dip and tip fold back toward the palm
    pip = base + d * seg * 0.4
    dip = pip - d * seg * 0.5
    tip = dip - d * seg * 0.6
    return [tuple(base), tuple(pip), tuple(dip), tuple(tip)]


def make_pose(*, thumb="curled", index=True, middle=True, ring=True, pinky=True) -> np.ndarray:
    """Build a synthetic 21-landmark hand with the requested finger states."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.50, 0.50, 0)                       # wrist
    lm[9] = (0.50, 0.40, 0)                       # middle MCP (palm size)

    fingers = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    bases = {"index": (0.44, 0.40), "middle": (0.50, 0.40),
             "ring": (0.56, 0.40), "pinky": (0.62, 0.42)}
    chain_ids = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
                 "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

    for name, ext in fingers.items():
        pts = _chain(bases[name], straight=ext)
        for idx, (x, y) in zip(chain_ids[name], pts):
            lm[idx] = (x, y, 0)

    for idx, (x, y) in zip((1, 2, 3, 4), THUMB_POSES[thumb]):
        lm[idx] = (x, y, 0)
    return lm


def test_open_palm():
    label, score = classify_static(make_pose(thumb="straight"))
    assert label == "Open_Palm" and score > 0.5


def test_closed_fist():
    label, score = classify_static(
        make_pose(thumb="curled", index=False, middle=False, ring=False, pinky=False))
    assert label == "Closed_Fist" and score > 0.5


def test_victory():
    label, _ = classify_static(make_pose(thumb="curled", ring=False, pinky=False))
    assert label == "Victory"


def test_pointing_up():
    label, _ = classify_static(
        make_pose(thumb="curled", middle=False, ring=False, pinky=False))
    assert label == "Pointing_Up"


def test_thumb_up_and_down():
    label, _ = classify_static(
        make_pose(thumb="up", index=False, middle=False, ring=False, pinky=False))
    assert label == "Thumb_Up"
    label, _ = classify_static(
        make_pose(thumb="down", index=False, middle=False, ring=False, pinky=False))
    assert label == "Thumb_Down"


def test_i_love_you():
    label, _ = classify_static(make_pose(thumb="straight", middle=False, ring=False))
    assert label == "ILoveYou"


def test_undefined_pose_returns_none():
    label, score = classify_static(make_pose(thumb="curled", pinky=False))
    assert label == NONE_LABEL and score == 0.0


def test_certainties_are_bounded():
    certs = finger_certainties(make_pose(thumb="straight"))
    assert set(certs) == {"thumb", "index", "middle", "ring", "pinky"}
    for v in certs.values():
        assert 0.0 <= v <= 1.0

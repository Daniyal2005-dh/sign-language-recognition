"""Unit tests for landmark normalisation and frame features (pure NumPy)."""
import numpy as np

from signai.config import FEATS_PER_HAND, FRAME_FEATURES
from signai.features import (
    frame_features,
    frame_has_hand,
    hand_features,
    normalize_hand,
    order_hands,
)


def canonical_hand() -> np.ndarray:
    """A plain 21x3 hand: wrist at (0.5, 0.5), palm size 0.1, fingers up."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.50, 0.50, 0)
    lm[9] = (0.50, 0.40, 0)                       # middle MCP defines palm size
    for base, xs in ((5, 0.44), (9, 0.50), (13, 0.56), (17, 0.62)):
        y0 = 0.40 if base != 17 else 0.42
        for k, idx in enumerate(range(base, base + 4)):
            lm[idx] = (xs, y0 - 0.035 * k, 0)
    # thumb: extended out to the left
    for idx, xy in zip((1, 2, 3, 4),
                       [(0.44, 0.47), (0.38, 0.44), (0.34, 0.41), (0.30, 0.38)]):
        lm[idx] = (*xy, 0)
    return lm


def test_normalize_hand_is_translation_and_scale_invariant():
    lm = canonical_hand()
    a = normalize_hand(lm)
    b = normalize_hand(lm * 2.0 + 0.1)            # zoom + pan
    assert np.allclose(a, b, atol=1e-5)
    assert np.allclose(a[0], 0)                   # wrist at origin
    assert abs(np.linalg.norm(a[9]) - 1.0) < 1e-5  # palm size == 1


def test_hand_features_shape():
    assert hand_features(canonical_hand()).shape == (FEATS_PER_HAND,)


def test_frame_features_orders_hands_by_wrist_x():
    h1 = canonical_hand()                         # wrist x = 0.50
    h2 = canonical_hand() + np.float32([0.2, 0, 0])  # wrist x = 0.70
    feats = frame_features([h2, h1])              # deliberately reversed input
    assert np.allclose(feats[:FEATS_PER_HAND], hand_features(h1), atol=1e-6)
    assert np.allclose(feats[FEATS_PER_HAND:], hand_features(h2), atol=1e-6)


def test_frame_features_pads_missing_hand_with_zeros():
    h1 = canonical_hand()
    feats = frame_features([h1])
    assert feats.shape == (FRAME_FEATURES,)
    assert np.any(feats[:FEATS_PER_HAND])          # hand present
    assert not np.any(feats[FEATS_PER_HAND:])      # second slot empty

    empty = frame_features([])
    assert empty.shape == (FRAME_FEATURES,) and not frame_has_hand(empty)
    assert frame_has_hand(feats)


def test_order_hands_returns_sorted_list():
    h1 = canonical_hand()
    h2 = canonical_hand() - np.float32([0.3, 0, 0])
    ordered = order_hands([h1, h2])
    assert ordered[0][0][0] < ordered[1][0][0]

"""Generate synthetic landmark data to exercise the training pipeline.

Useful for verifying train.py end-to-end (and in CI) without a webcam:
every class gets a distinct random hand-pose template, and each sample adds
smooth per-frame motion around that template.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signai.config import (  # noqa: E402
    DEFAULT_SEQ_LEN,
    FRAME_FEATURES,
    NUM_COORDS,
    NUM_LANDMARKS,
    STATIC_DIR,
    SEQUENCES_DIR,
)
from signai.features import frame_features  # noqa: E402


def make_template(rng: np.random.Generator) -> np.ndarray:
    """A random but palm-sized hand pose, wrist at origin."""
    lm = rng.normal(0.0, 0.22, size=(NUM_LANDMARKS, NUM_COORDS)).astype(np.float32)
    lm[0] = 0.0
    lm[:, 2] *= 0.25
    return lm


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic landmark samples")
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--samples-per-class", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prefix", default="class")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = SEQUENCES_DIR if args.mode == "dynamic" else STATIC_DIR

    for ci in range(args.classes):
        label = f"{args.prefix}{ci + 1}"
        template = make_template(rng)
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)

        for si in range(args.samples_per_class):
            if args.mode == "dynamic":
                drift = rng.normal(0.0, 0.012, size=(args.seq_len, NUM_LANDMARKS, NUM_COORDS))
                drift = np.cumsum(drift, axis=0) * 0.4                  # smooth motion
                jitter = rng.normal(0.0, 0.015, size=(args.seq_len, NUM_LANDMARKS, NUM_COORDS))
                seq = template[None, :, :] + drift + jitter             # (T, 21, 3)
                feats = np.stack([frame_features([seq[t]]) for t in range(args.seq_len)])
            else:
                jitter = rng.normal(0.0, 0.015, size=(NUM_LANDMARKS, NUM_COORDS))
                feats = frame_features([template + jitter])
            np.savez_compressed(
                class_dir / f"{label}_synthetic_{si:04d}.npz",
                features=feats.astype(np.float32),
                label=label,
                signer="synthetic",
                mode=args.mode,
                recorded="synthetic",
            )
        print(f"  wrote {args.samples_per_class} samples -> {class_dir}")

    print(f"\nDone. Train with:\n  python train.py --mode {args.mode}")


if __name__ == "__main__":
    main()

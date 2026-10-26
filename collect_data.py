"""Record labelled hand-landmark samples from the webcam for training.

Dynamic mode records ``--seq-len`` consecutive frames (~1 second of signing);
static mode records a single averaged frame. Samples are saved as ``.npz``
files named ``<label>_<signer>_<idx>.npz`` so training can split by signer.

Examples
--------
    python collect_data.py --label hello
    python collect_data.py --label thankyou --samples 30 --signer daniyal
    python collect_data.py --label A --mode static --samples 40

Keys: SPACE = skip countdown, Q/ESC = stop early.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from signai.config import DEFAULT_SEQ_LEN, STATIC_DIR, SEQUENCES_DIR  # noqa: E402
from signai.features import frame_features, frame_has_hand  # noqa: E402
from signai.hands import HandPipeline, draw_hand_frame  # noqa: E402

WINDOW = "SignAI data collection"


def draw_banner(img, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def next_index(class_dir: Path, label: str, signer: str) -> int:
    pattern = f"{label}_{signer}_*.npz"
    existing = list(class_dir.glob(pattern))
    if not existing:
        return 0
    return max(int(p.stem.rsplit("_", 1)[1]) for p in existing) + 1


def read_frame(cap, pipeline):
    ok, img = cap.read()
    if not ok:
        return None, None
    img = cv2.flip(img, 1)                       # mirror for a natural "selfie" view
    frame = pipeline.process_bgr(img)
    return img, frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect hand-landmark training data")
    ap.add_argument("--label", required=True, help="class name, e.g. hello, thankyou, A")
    ap.add_argument("--mode", choices=["dynamic", "static"], default="dynamic",
                    help="dynamic = sequence of frames (movement signs), static = one pose")
    ap.add_argument("--samples", type=int, default=20, help="samples to record this run")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="frames per dynamic sample")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--signer", default="me", help="who is signing (used to split data later)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()

    root = SEQUENCES_DIR if args.mode == "dynamic" else STATIC_DIR
    class_dir = root / args.label
    class_dir.mkdir(parents=True, exist_ok=True)

    pipeline = HandPipeline(num_hands=2, use_task_model=False)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    idx = next_index(class_dir, args.label, args.signer)
    print(f"Collecting '{args.label}' ({args.mode}) for signer '{args.signer}' -> {class_dir}")
    print("Keys: SPACE = skip countdown | Q / ESC = stop\n")

    saved = 0
    try:
        while saved < args.samples:
            # ------------------------------------------------ countdown
            t0 = time.time()
            skip = False
            while time.time() - t0 < 3.0:
                img, frame = read_frame(cap, pipeline)
                if img is None:
                    sys.exit("Camera read failed")
                secs = max(0.0, 3.0 - (time.time() - t0))
                img = draw_hand_frame(img, frame,
                                      text=f"Get ready: '{args.label}' in {secs:.1f}s")
                draw_banner(img, [
                    f"Sample {saved + 1}/{args.samples}  |  label={args.label}  signer={args.signer}",
                    "Hold the sign clearly in view. SPACE skips the countdown.",
                ])
                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    raise KeyboardInterrupt
                if key == ord(" "):
                    skip = True
                    break
            if skip:
                time.sleep(0.2)

            # ------------------------------------------------ record
            if args.mode == "dynamic":
                buffer: list[np.ndarray] = []
                while len(buffer) < args.seq_len:
                    img, frame = read_frame(cap, pipeline)
                    if img is None:
                        sys.exit("Camera read failed")
                    feats = frame_features([h.landmarks for h in frame.hands])
                    if frame_has_hand(feats):              # only record frames with a hand
                        buffer.append(feats)
                    done = len(buffer) / args.seq_len
                    cv2.rectangle(img, (12, 12), (12 + int(300 * done), 34), (80, 220, 100), -1)
                    cv2.rectangle(img, (12, 12), (312, 34), (255, 255, 255), 1)
                    img = draw_hand_frame(img, frame,
                                          text=f"REC '{args.label}' {len(buffer)}/{args.seq_len}")
                    draw_banner(img, [f"Recording sample {saved + 1}/{args.samples}"])
                    cv2.imshow(WINDOW, img)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        raise KeyboardInterrupt
                features = np.stack(buffer).astype(np.float32)        # (T, 126)
            else:
                acc: list[np.ndarray] = []
                while len(acc) < 10:                                   # average 10 frames
                    img, frame = read_frame(cap, pipeline)
                    if img is None:
                        sys.exit("Camera read failed")
                    feats = frame_features([h.landmarks for h in frame.hands])
                    if frame_has_hand(feats):              # only store frames with a hand
                        acc.append(feats)
                    img = draw_hand_frame(img, frame, text=f"Static pose '{args.label}'")
                    draw_banner(img, [f"Hold the pose... sample {saved + 1}/{args.samples}"])
                    cv2.imshow(WINDOW, img)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        raise KeyboardInterrupt
                features = np.mean(np.stack(acc), axis=0).astype(np.float32)   # (126,)

            # ------------------------------------------------ save
            out = class_dir / f"{args.label}_{args.signer}_{idx:04d}.npz"
            np.savez_compressed(
                out,
                features=features,
                label=args.label,
                signer=args.signer,
                mode=args.mode,
                recorded=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            idx += 1
            saved += 1
            print(f"  saved {out.name}  shape={features.shape}")

            img, frame = read_frame(cap, pipeline)
            if img is not None:
                img = draw_hand_frame(img, frame, text=f"Saved {saved}/{args.samples}!")
                cv2.imshow(WINDOW, img)
                cv2.waitKey(700)                                       # brief pause between samples
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()

    total = sum(1 for _ in class_dir.glob("*.npz"))
    print(f"\nDone. '{args.label}' now has {total} sample file(s) in {class_dir}")
    print("Collect at least 2 classes, then train:\n  python train.py --mode " + args.mode)


if __name__ == "__main__":
    main()

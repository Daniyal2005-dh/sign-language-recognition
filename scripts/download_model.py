"""Download the MediaPipe gesture recognizer model into models/.

The file is committed to the repository so the cloud deployment works
out of the box; this script re-downloads it if it is ever missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signai.config import GESTURE_MODEL_PATH  # noqa: E402
from signai.hands import ensure_gesture_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the MediaPipe gesture model")
    ap.add_argument("--force", action="store_true", help="re-download even if it exists")
    args = ap.parse_args()

    if args.force and GESTURE_MODEL_PATH.exists():
        GESTURE_MODEL_PATH.unlink()

    ok = ensure_gesture_model()
    if ok:
        size_mb = GESTURE_MODEL_PATH.stat().st_size / 1e6
        print(f"OK: {GESTURE_MODEL_PATH} ({size_mb:.1f} MB)")
    else:
        sys.exit("Download failed - check your internet connection, or "
                 "the app will fall back to the rule-based classifier.")


if __name__ == "__main__":
    main()

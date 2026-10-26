"""Quick local demo: OpenCV window + offline speech (no browser, no Streamlit).

Useful as a fast sanity check of the full recognition loop on your machine.

Keys: Q = quit | C = clear sentence | S = speak sentence | D = toggle drawing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from signai.config import word_for_gesture  # noqa: E402
from signai.hands import HandPipeline, draw_hand_frame  # noqa: E402
from signai.predict import WordCommitter  # noqa: E402
from signai.speech import pyttsx3_speak  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SignAI local demo (OpenCV window)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--rules-only", action="store_true",
                    help="skip the pretrained gesture model, use geometry rules only")
    ap.add_argument("--speak", action="store_true", help="speak each recognised word (pyttsx3)")
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--stable", type=int, default=6, help="frames a gesture must hold")
    ap.add_argument("--cooldown", type=float, default=1.5, help="seconds between same-word commits")
    args = ap.parse_args()

    pipeline = HandPipeline(num_hands=2, use_task_model=not args.rules_only)
    committer = WordCommitter(stable_frames=args.stable, cooldown_s=args.cooldown,
                              min_score=args.min_score)
    sentence: list[str] = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open camera {args.camera}")

    fps, t_prev, draw = 0.0, time.time(), True
    print("Keys: Q quit | C clear | S speak sentence\n")

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            img = cv2.flip(img, 1)
            frame = pipeline.process_bgr(img)

            label, score, hand = "", 0.0, frame.best_hand()
            if hand is not None:
                label, score = hand.gesture, hand.score
            word = committer.update(label, score)
            if word:
                spoken = word_for_gesture(word)
                if spoken:
                    sentence.append(spoken)
                    print("committed:", spoken)
                    if args.speak:
                        pyttsx3_speak(spoken)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            shown = word_for_gesture(label) or "-"
            if draw:
                img = draw_hand_frame(img, frame, text=shown, conf=score, fps=fps)
            cv2.putText(img, " ".join(sentence)[-70:], (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 220, 25), 2, cv2.LINE_AA)
            cv2.putText(img, "[Q]quit [C]clear [S]speak [D]draw", (12, img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("SignAI local demo", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("c"):
                sentence.clear()
            if key == ord("s"):
                pyttsx3_speak(" ".join(sentence))
            if key == ord("d"):
                draw = not draw
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()

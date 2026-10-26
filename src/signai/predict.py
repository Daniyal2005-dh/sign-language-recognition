"""Turn a noisy per-frame gesture stream into discrete words and load models.

Two pieces:

* :class:`WordCommitter` - a small state machine: a gesture label must survive
  ``stable_frames`` consecutive frames, pass a cooldown, and follow a small
  "gap" (no-hand frames) before the same word is emitted again. This converts
  the continuous stream into one clean word per deliberate sign.
* :class:`SequencePredictor` - runs a custom-trained dynamic model (LSTM /
  Transformer) over a sliding window of landmark sequences. TensorFlow is
  imported lazily so the deployed app works without it.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import DEFAULT_SEQ_LEN, FRAME_FEATURES
from .gestures import NONE_LABEL


class WordCommitter:
    """Accumulate per-frame (label, score) and emit discrete words."""

    def __init__(
        self,
        stable_frames: int = 6,
        cooldown_s: float = 1.5,
        gap_frames: int = 3,
        min_score: float = 0.6,
    ) -> None:
        self.stable_frames = int(stable_frames)
        self.cooldown_s = float(cooldown_s)
        self.gap_frames = int(gap_frames)
        self.min_score = float(min_score)

        self.candidate: str | None = None
        self.count: int = 0
        self._gap: int = 0
        self._armed: bool = True
        self._last_word: str | None = None
        self._last_time: float = float("-inf")

    def reset(self) -> None:
        self.__dict__.update(type(self)().__dict__)

    def update(self, label: str, score: float, now: float | None = None) -> str | None:
        """Feed one frame's prediction. Returns the committed word, or None."""
        now = time.time() if now is None else now

        if label == NONE_LABEL or score < self.min_score:
            self._gap += 1
            self.candidate, self.count = None, 0
            if self._gap >= self.gap_frames:
                self._armed = True
            return None

        self._gap = 0
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1

        can_commit = self.count >= self.stable_frames and now - self._last_time >= self.cooldown_s
        # After committing a word, require a hand "gap" before the SAME word can
        # commit again; a DIFFERENT word may commit right away.
        same_word_block = self._last_word is not None and label == self._last_word and not self._armed

        if can_commit and not same_word_block:
            self._armed = False
            self._last_word = label
            self._last_time = now
            self.count = 0
            return label

        return None


class SequencePredictor:
    """Sliding-window classifier for custom-trained dynamic (LSTM) models."""

    def __init__(self, model_path: Path | str, labels_path: Path | str) -> None:
        import tensorflow as tf  # lazy: only needed for custom models

        self.model = tf.keras.models.load_model(str(model_path))
        meta = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.labels: list[str] = meta["labels"]
        self.mode: str = meta.get("mode", "dynamic")
        self.seq_len: int = int(meta.get("seq_len", DEFAULT_SEQ_LEN))
        self._buffer: deque = deque(maxlen=self.seq_len)

    def add_frame(self, feats: np.ndarray) -> tuple[str, float]:
        """Feed one (126,) frame feature vector; get (label, score).

        Returns ("", 0.0) while the window is still filling up.
        """
        self._buffer.append(np.asarray(feats, dtype=np.float32).reshape(FRAME_FEATURES))
        if len(self._buffer) < self.seq_len:
            return "", 0.0
        x = np.stack(self._buffer)[None, ...]  # (1, T, F)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def reset(self) -> None:
        self._buffer.clear()

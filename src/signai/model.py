"""Keras model builders: LSTM, Transformer encoder, and MLP (static poses).

The dynamic models consume sequences of shape ``(seq_len, 126)`` (frames of
two-hand landmark features). The ``Masking(0.0)`` layer lets the model ignore
the zero-padded hand slot and zero-padded trailing frames.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import FRAME_FEATURES


# --------------------------------------------------------------- dynamic models
def build_lstm(n_classes: int, seq_len: int, n_feats: int = FRAME_FEATURES, units: int = 64):
    """Bidirectional-free, regularised stacked LSTM - the reference model."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(seq_len, n_feats), name="landmarks")
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(units)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="label")(x)
    return keras.Model(inp, out, name="signai_lstm")


def build_transformer(n_classes: int, seq_len: int, n_feats: int = FRAME_FEATURES,
                      embed_dim: int = 64, heads: int = 4, blocks: int = 2):
    """A small Transformer encoder over frame features (spec alternative)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(seq_len, n_feats), name="landmarks")
    x = layers.Dense(embed_dim)(inp)
    pos_cls = _pos_embedding_cls()
    x = pos_cls(seq_len, embed_dim, name="position_embedding")(x)
    for _ in range(blocks):
        # self-attention sub-layer
        h = layers.LayerNormalization(epsilon=1e-5)(x)
        h = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim // heads,
                                      dropout=0.1)(h, h)
        x = layers.Add()([x, h])
        # feed-forward sub-layer
        h = layers.LayerNormalization(epsilon=1e-5)(x)
        h = layers.Dense(embed_dim * 2, activation="gelu")(h)
        h = layers.Dropout(0.2)(h)
        h = layers.Dense(embed_dim)(h)
        x = layers.Add()([x, h])
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="label")(x)
    return keras.Model(inp, out, name="signai_transformer")


# ---------------------------------------------------------------- static model
def build_static_mlp(n_classes: int, n_feats: int = FRAME_FEATURES, hidden=(128, 64)):
    """Feed-forward classifier for single-frame (static) poses."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(n_feats,), name="landmarks")
    x = inp
    for units in hidden:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="label")(x)
    return keras.Model(inp, out, name="signai_mlp")


# ------------------------------------------------------------ position layer
def _pos_embedding_cls():
    """Position-embedding layer, registered for Keras serialization."""
    from tensorflow import keras

    @keras.saving.register_keras_serializable(package="signai")
    class AddPositionEmbedding(keras.layers.Layer):
        def __init__(self, seq_len: int, embed_dim: int, **kwargs):
            super().__init__(**kwargs)
            self.seq_len = seq_len
            self.embed_dim = embed_dim

        def build(self, input_shape):
            self.pos = self.add_weight(
                name="pos",
                shape=(1, self.seq_len, self.embed_dim),
                initializer="random_normal",
                trainable=True,
            )

        def call(self, x):
            return x + self.pos

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"seq_len": self.seq_len, "embed_dim": self.embed_dim})
            return cfg

    return AddPositionEmbedding


# --------------------------------------------------------------- persistence
def save_model(model, models_dir: Path, name: str, labels: list[str], extra: dict | None = None) -> Path:
    """Save ``<name>.keras`` plus a sibling ``<name>.labels.json``."""
    from tensorflow import keras

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.keras"
    model.save(model_path)

    meta = {
        "labels": labels,
        "frame_features": FRAME_FEATURES,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta.update(extra or {})
    (models_dir / f"{name}.labels.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return model_path


def compile_model(model, lr: float = 1e-3):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

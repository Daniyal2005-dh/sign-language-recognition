"""Load collected landmark samples from disk and split into train/val/test.

File layout produced by ``collect_data.py``::

    data/sequences/<label>/<label>_<signer>_<idx>.npz   dynamic: (T, 126)
    data/static/<label>/<label>_<signer>_<idx>.npz      static:  (126,)

Each ``.npz`` stores the ``features`` array plus metadata keys (``label``,
``signer``, ``mode``). Recording the signer id in the filename lets
``split(..., group_by_signer=True)`` hold out entire signers from the test set,
which is the honest way to measure how well the model generalises.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import FRAME_FEATURES, STATIC_DIR, SEQUENCES_DIR


def dataset_dir(mode: str) -> Path:
    return STATIC_DIR if mode == "static" else SEQUENCES_DIR


def list_classes(root: Path) -> list[tuple[str, list[Path]]]:
    """Sorted [(class_name, [npz files])] for every class folder with data."""
    classes: list[tuple[str, list[Path]]] = []
    if not root.exists():
        return classes
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(d.glob("*.npz"))
        if files:
            classes.append((d.name, files))
    return classes


def _fix_len(feats: np.ndarray, seq_len: int) -> np.ndarray:
    """Pad with zeros or truncate a (T, F) sequence to exactly seq_len rows."""
    t = feats.shape[0]
    if t == seq_len:
        return feats
    if t > seq_len:
        return feats[:seq_len]
    pad = np.zeros((seq_len - t, feats.shape[1]), dtype=feats.dtype)
    return np.concatenate([feats, pad], axis=0)


def load_dataset(root: Path, seq_len: int | None = None):
    """Load all samples under ``root``.

    Returns ``(X, y, signers, class_names)`` where X is (N, T, F) for dynamic
    data or (N, F) for static data.
    """
    classes = list_classes(root)
    if not classes:
        raise FileNotFoundError(
            f"No samples found under {root}. Collect data first, e.g.:\n"
            f"  python collect_data.py --label <name>"
        )

    X: list[np.ndarray] = []
    y: list[int] = []
    signers: list[str] = []

    for class_idx, (name, files) in enumerate(classes):
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                feats = np.asarray(z["features"], dtype=np.float32)
                signer = str(z["signer"].item()) if "signer" in z.files else "unknown"
            if seq_len is not None and feats.ndim == 2:
                feats = _fix_len(feats, seq_len)
            X.append(feats)
            y.append(class_idx)
            signers.append(signer)

    sizes = {x.size for x in X}
    if len(sizes) != 1:
        raise ValueError(
            "Samples have inconsistent lengths. Dynamic samples in one dataset "
            "must share the same sequence length - re-collect with a fixed "
            "--seq-len, or pass --seq-len to pad/truncate."
        )
    size = sizes.pop()

    if size == FRAME_FEATURES and X[0].ndim <= 2:
        X = np.stack([x.reshape(FRAME_FEATURES) for x in X])          # (N, 126)
    elif size % FRAME_FEATURES == 0 and X[0].ndim == 2:
        X = np.stack([x.reshape(-1, FRAME_FEATURES) for x in X])      # (N, T, 126)
    else:
        raise ValueError(f"Unexpected feature size {size}; expected multiples of {FRAME_FEATURES}")

    return X, np.asarray(y), signers, [name for name, _ in classes]


def split_dataset(
    y: np.ndarray,
    signers: list[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    group_by_signer: bool = False,
    seed: int = 42,
):
    """Split into train/val/test index lists.

    ``group_by_signer=True`` holds out whole signers (recommended whenever the
    dataset contains >= 3 signers) so test performance reflects unseen people.
    Otherwise a stratified per-class split is used.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)

    if group_by_signer:
        unique = sorted(set(signers))
        if len(unique) >= 3:
            rng.shuffle(unique)
            n_test = max(1, int(round(len(unique) * test_frac)))
            n_val = max(1, int(round(len(unique) * val_frac)))
            test_signers = set(unique[:n_test])
            val_signers = set(unique[n_test : n_test + n_val])
            train_idx = [i for i in idx if signers[i] not in test_signers and signers[i] not in val_signers]
            val_idx = [i for i in idx if signers[i] in val_signers]
            test_idx = [i for i in idx if signers[i] in test_signers]
            return list(train_idx), list(val_idx), list(test_idx)
        # too few signers: fall through to stratified split

    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        cls_idx = idx[y[idx] == cls]
        n_test = int(round(len(cls_idx) * test_frac))
        n_val = int(round(len(cls_idx) * val_frac))
        test_idx += list(cls_idx[:n_test])
        val_idx += list(cls_idx[n_test : n_test + n_val])
        train_idx += list(cls_idx[n_test + n_val :])

    return train_idx, val_idx, test_idx

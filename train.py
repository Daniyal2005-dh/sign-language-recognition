"""Train a sign classifier on collected landmark data.

Dynamic data  -> LSTM (default) or Transformer encoder over (T, 126) sequences.
Static data   -> MLP over (126,) single frames.

Outputs
-------
    models/<name>.keras + models/<name>.labels.json
    reports/<name>_confusion_matrix.png
    reports/<name>_history.png
    reports/<name>_metrics.json

Examples
--------
    python train.py                                   # LSTM on data/sequences
    python train.py --arch transformer --epochs 80
    python train.py --mode static                     # MLP on data/static
    python train.py --group-by-signer                 # hold out whole signers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from signai.config import MODELS_DIR, REPORTS_DIR  # noqa: E402
from signai.dataset import dataset_dir, load_dataset, split_dataset  # noqa: E402
from signai.model import (  # noqa: E402
    build_lstm,
    build_static_mlp,
    build_transformer,
    compile_model,
    save_model,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the SignAI gesture classifier")
    ap.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--arch", choices=["lstm", "transformer", "mlp"], default=None,
                    help="default: lstm (dynamic) / mlp (static)")
    ap.add_argument("--name", default=None, help="output model name (default sign_<mode>_<arch>)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override/pad-truncate sequence length (dynamic only)")
    ap.add_argument("--group-by-signer", action="store_true",
                    help="split by signer (recommended with >= 3 signers)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    arch = args.arch or ("mlp" if args.mode == "static" else "lstm")
    if args.mode == "static" and arch != "mlp":
        sys.exit("Static data is single-frame: use --arch mlp.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report
    from tensorflow import keras

    root = dataset_dir(args.mode)
    X, y, signers, class_names = load_dataset(root, seq_len=args.seq_len)
    n_classes = len(class_names)

    counts = {name: int((y == i).sum()) for i, name in enumerate(class_names)}
    print(f"Dataset: {X.shape} | classes: {n_classes} | samples per class: {counts}")
    if n_classes < 2:
        sys.exit("Need at least 2 classes. Collect data for another sign first.")
    weak = [f"{k} ({v})" for k, v in counts.items() if v < 5]
    if weak:
        print(f"WARNING: very few samples for {', '.join(weak)} - expect poor accuracy.")

    tr, va, te = split_dataset(
        y, signers, group_by_signer=args.group_by_signer, seed=args.seed,
    )
    if not tr or not va or not te:
        sys.exit("Not enough samples to split. Collect more data per class.")
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}"
          + (" (by signer)" if args.group_by_signer else ""))

    seq_len = X.shape[1] if X.ndim == 3 else 0
    if arch == "lstm":
        model = build_lstm(n_classes, seq_len, X.shape[2])
    elif arch == "transformer":
        model = build_transformer(n_classes, seq_len, X.shape[2])
    else:
        model = build_static_mlp(n_classes, X.shape[1])
    compile_model(model, lr=args.lr)
    model.summary(print_fn=print)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=6, min_lr=1e-5),
    ]
    history = model.fit(
        X[tr], y[tr],
        validation_data=(X[va], y[va]),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2,
    )

    # ------------------------------------------------------------- evaluation
    probs = model.predict(X[te], verbose=0)
    pred = probs.argmax(axis=1)
    report = classification_report(y[te], pred, target_names=class_names,
                                   digits=3, zero_division=0)
    print("\nTest-set report:\n" + report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or f"sign_{args.mode}_{arch}"

    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * n_classes, 1.0 + 0.8 * n_classes))
    ConfusionMatrixDisplay.from_predictions(y[te], pred, display_labels=class_names,
                                            normalize="true", cmap="Blues",
                                            colorbar=False, ax=ax)
    ax.set_title(f"{name} - confusion matrix (test, normalised)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{name}_history.png", dpi=150)
    plt.close(fig)

    test_acc = float((pred == y[te]).mean())
    val_acc = float(max(history.history["val_accuracy"]))
    (REPORTS_DIR / f"{name}_metrics.json").write_text(json.dumps({
        "name": name, "mode": args.mode, "arch": arch,
        "n_classes": n_classes, "class_names": class_names,
        "counts": counts, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "val_accuracy": val_acc, "test_accuracy": test_acc,
        "group_by_signer": args.group_by_signer,
    }, indent=2))

    # ----------------------------------------------------------------- saving
    extra = {"mode": args.mode, "arch": arch, "seq_len": int(seq_len) or None,
             "val_accuracy": val_acc, "test_accuracy": test_acc}
    path = save_model(model, MODELS_DIR, name, class_names, extra=extra)
    print(f"\nSaved model  -> {path}")
    print(f"Saved labels -> {path.parent / (name + '.labels.json')}")
    print(f"Reports      -> {REPORTS_DIR}\\{name}_*.png / _metrics.json")
    print("\nNext: in the Streamlit app choose 'My trained model' (sidebar) and select this model.")


if __name__ == "__main__":
    main()

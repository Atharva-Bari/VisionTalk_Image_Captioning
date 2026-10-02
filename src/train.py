"""Train the CNN + LSTM captioner. Importing this module does not start training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FEATURE_VECTOR_DIM,
    FEATURES_METADATA_PATH,
    FINAL_MODEL_PATH,
    HISTORY_PATH,
    LEARNING_RATE,
    LOSS_PLOT_PATH,
    METADATA_PATH,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SEQUENCES_PATH,
    SPLIT_PATH,
    TOKENIZER_PATH,
    ensure_project_directories,
)
from src.data_generator import make_tf_dataset
from src.model import build_caption_model
from src.tokenizer_utils import load_tokenizer
from src.utils import dump_json, load_json, set_global_seed, setup_logging


def _load_training_artifacts() -> tuple[dict, dict, dict, dict]:
    for path, hint in (
        (TOKENIZER_PATH, "python -m src.text_preprocessing"),
        (SEQUENCES_PATH, "python -m src.text_preprocessing"),
        (SPLIT_PATH, "python -m src.text_preprocessing"),
        (METADATA_PATH, "python -m src.text_preprocessing"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}. Run: {hint}")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    sequences = load_json(SEQUENCES_PATH)
    splits = load_json(SPLIT_PATH)
    metadata = load_json(METADATA_PATH)
    return tokenizer, sequences, splits, metadata


def _feature_dim() -> int:
    if FEATURES_METADATA_PATH.is_file():
        meta = load_json(FEATURES_METADATA_PATH)
        dim = meta.get("feature_vector_dim")
        if dim is not None:
            return int(dim)
    return FEATURE_VECTOR_DIM


def _plot_history(history: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if "loss" in history:
        plt.plot(history["loss"], label="train loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Sparse categorical cross-entropy")
    plt.title("VisionTalk training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    resume: bool = False,
    learning_rate: float = LEARNING_RATE,
) -> dict:
    """Train from scratch or resume from the latest checkpoint."""
    logger = setup_logging()
    ensure_project_directories()
    set_global_seed()

    tokenizer, sequences, splits, metadata = _load_training_artifacts()
    vocab_size = int(tokenizer["vocab_size"])
    max_length = int(tokenizer.get("max_caption_length") or metadata["max_caption_length"])
    feature_dim = _feature_dim()

    train_ds, train_steps, train_ids = make_tf_dataset(
        splits["train"],
        sequences,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=batch_size,
        feature_dim=feature_dim,
        shuffle=True,
    )
    val_ds, val_steps, val_ids = make_tf_dataset(
        splits["val"],
        sequences,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=batch_size,
        feature_dim=feature_dim,
        shuffle=False,
    )

    logger.info(
        "Training images=%s val images=%s vocab=%s max_len=%s feature_dim=%s "
        "steps/epoch=%s val_steps=%s epochs=%s (raise EPOCHS in src/config.py to 20–50 later)",
        len(train_ids),
        len(val_ids),
        vocab_size,
        max_length,
        feature_dim,
        train_steps,
        val_steps,
        epochs,
    )

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.models import load_model

    resume_path = CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else BEST_CHECKPOINT_PATH
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume: no checkpoint at {CHECKPOINT_PATH} or {BEST_CHECKPOINT_PATH}."
            )
        logger.info("Resuming from %s", resume_path)
        model = load_model(resume_path)
        if list(model.output_shape)[-1] != vocab_size:
            raise ValueError(
                f"Checkpoint output size {model.output_shape[-1]} does not match "
                f"tokenizer vocab_size {vocab_size}."
            )
    else:
        model = build_caption_model(
            vocab_size=vocab_size,
            max_length=max_length,
            feature_dim=feature_dim,
            learning_rate=learning_rate,
        )
        model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=str(BEST_CHECKPOINT_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    history_data = {key: [float(v) for v in values] for key, values in history.history.items()}
    dump_json(HISTORY_PATH, history_data)
    _plot_history(history_data, LOSS_PLOT_PATH)

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    logger.info("Saved final model to %s", FINAL_MODEL_PATH)
    logger.info("Best checkpoint (val_loss): %s", BEST_CHECKPOINT_PATH)
    logger.info("Latest checkpoint: %s", CHECKPOINT_PATH)
    return history_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VisionTalk CNN + LSTM")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Default is 3 for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true", help="Continue from the latest checkpoint.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    try:
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=args.resume,
            learning_rate=args.learning_rate,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

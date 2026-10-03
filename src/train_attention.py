"""Train the spatial-attention VGG16 + LSTM captioner on Flickr8k."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import (
    BATCH_SIZE, DATA_FEATURES_DIR, EARLY_STOPPING_PATIENCE, EPOCHS, FINAL_MODEL_DIR,
    HISTORY_PATH, IMAGE_EXTENSIONS, LEARNING_RATE, METADATA_PATH, REGION_FEATURES_DIR,
    SEQUENCES_PATH, SPLIT_PATH, TOKENIZER_PATH, ensure_project_directories,
    resolve_images_dir,
)
from src.attention_model import build_attention_caption_model
from src.region_data_generator import make_region_tf_dataset
from src.region_feature_extraction import extract_region_cache
from src.tokenizer_utils import load_tokenizer
from src.utils import dump_json, load_json, set_global_seed, setup_logging

ROOT = Path(__file__).resolve().parents[1]
ATTENTION_BEST_PATH = ROOT / "models" / "checkpoints" / "best_attention_caption_model.keras"
ATTENTION_LATEST_PATH = ROOT / "models" / "checkpoints" / "latest_attention_caption_model.keras"
ATTENTION_FINAL_PATH = FINAL_MODEL_DIR / "attention_caption_model.keras"
ATTENTION_HISTORY_PATH = ROOT / "outputs" / "attention_training_history.json"


def extract_features(force: bool = False, batch_size: int = 16) -> dict:
    """Cache 7x7 VGG16 region features for every labeled Flickr8k image."""
    ensure_project_directories()
    splits = load_json(SPLIT_PATH)
    ids = sorted({name for part in ("train", "val", "test") for name in splits[part]})
    images_dir = resolve_images_dir()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Flickr8k images were not found under {images_dir}")
    return extract_region_cache(ids, images_dir, REGION_FEATURES_DIR,
                                batch_size=batch_size, force=force)


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
          learning_rate: float = LEARNING_RATE) -> dict:
    logger = setup_logging()
    ensure_project_directories()
    set_global_seed()
    for path in (TOKENIZER_PATH, SEQUENCES_PATH, SPLIT_PATH, METADATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required preprocessing artifact missing: {path}. Run src.text_preprocessing first.")
    if not any(REGION_FEATURES_DIR.glob("*.npy")):
        raise FileNotFoundError("Spatial features are missing. Run: python -m src.train_attention --extract-only")

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    sequences = load_json(SEQUENCES_PATH)
    splits = load_json(SPLIT_PATH)
    vocab_size = int(tokenizer["vocab_size"])
    max_length = int(tokenizer["max_caption_length"])
    train_ds, train_steps, train_ids = make_region_tf_dataset(
        splits["train"], sequences, max_length, vocab_size, batch_size=batch_size,
        shuffle=True,
    )
    val_ds, val_steps, val_ids = make_region_tf_dataset(
        splits["val"], sequences, max_length, vocab_size, batch_size=batch_size,
        shuffle=False,
    )
    logger.info("Spatial attention training: train=%d val=%d steps=%d val_steps=%d epochs=%d",
                len(train_ids), len(val_ids), train_steps, val_steps, epochs)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    model = build_attention_caption_model(vocab_size, max_length, learning_rate=learning_rate)
    model.summary()
    callbacks = [
        ModelCheckpoint(str(ATTENTION_BEST_PATH), monitor="val_loss", save_best_only=True, verbose=1),
        ModelCheckpoint(str(ATTENTION_LATEST_PATH), monitor="val_loss", save_best_only=False),
        EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]
    history = model.fit(train_ds, epochs=epochs, steps_per_epoch=train_steps,
                        validation_data=val_ds, validation_steps=val_steps,
                        callbacks=callbacks, verbose=1)
    data = {key: [float(value) for value in values] for key, values in history.history.items()}
    dump_json(ATTENTION_HISTORY_PATH, data)
    ATTENTION_FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(ATTENTION_FINAL_PATH)
    logger.info("Saved best-restored attention model to %s", ATTENTION_FINAL_PATH)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--extract-only", action="store_true", help="Build the resumable spatial feature cache and exit.")
    parser.add_argument("--force-features", action="store_true", help="Recompute cached region features.")
    args = parser.parse_args()
    try:
        if args.extract_only:
            print(json.dumps(extract_features(force=args.force_features), indent=2))
        else:
            train(args.epochs, args.batch_size, args.learning_rate)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

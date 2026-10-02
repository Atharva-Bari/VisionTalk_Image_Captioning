"""VisionTalk CLI. Each operation is explicit; nothing runs as a hidden pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from src.config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    CHECKPOINTS_DIR,
    CNN_MODEL_NAME,
    DATA_FEATURES_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    EPOCHS,
    FINAL_MODEL_DIR,
    FINAL_MODEL_PATH,
    IMAGE_EXTENSIONS,
    METADATA_PATH,
    PROCESSED_CAPTIONS_PATH,
    PROJECT_ROOT,
    dataset_missing_message,
    ensure_project_directories,
    resolve_caption_file,
    resolve_images_dir,
)
from src.text_preprocessing import artifacts_exist, print_dataset_statistics, run_preprocessing


def _print_menu() -> None:
    print("VisionTalk – Image Captioning using CNN + LSTM")
    print("Available operations:")
    print("  1. python main.py preprocess          Preprocess captions / tokenizer")
    print("  2. python main.py extract-features    Extract CNN features (add --limit 8 to test)")
    print("  3. python main.py train               Train CNN + LSTM (default EPOCHS=3)")
    print("  4. python main.py generate --image X  Generate a caption")
    print("  5. python main.py evaluate            BLEU-1 / BLEU-4 on the test split")
    print("  6. python main.py status              Show dataset / artifact status")
    print()
    print("Module form (same operations):")
    print("  python -m src.text_preprocessing")
    print("  python -m src.feature_extraction")
    print("  python -m src.train")
    print("  python -m src.inference --image path\\to\\image.jpg")
    print("  python -m src.evaluation")
    print("  streamlit run app/streamlit_app.py")


def _print_status() -> None:
    ensure_project_directories()
    images_dir = resolve_images_dir()
    captions_file = resolve_caption_file()
    print("VisionTalk - Image Captioning")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data:     {DATA_RAW_DIR}")
    print(f"Processed:    {DATA_PROCESSED_DIR}")
    print(f"Features:     {DATA_FEATURES_DIR}")
    print(f"Checkpoints:  {CHECKPOINTS_DIR}")
    print(f"Final model:  {FINAL_MODEL_DIR}")
    print(f"CNN encoder:  {CNN_MODEL_NAME}")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Epochs:       {EPOCHS}  (smoke-test default; use 20–50 after the pipeline works)")
    print()
    print(f"Resolved images:   {images_dir}")
    print(f"Resolved captions: {captions_file}")
    print(f"Feature vectors:   {DATA_FEATURES_DIR / 'vectors'}")
    print(f"Best checkpoint:   {BEST_CHECKPOINT_PATH}")
    print(f"Final model file:  {FINAL_MODEL_PATH}")
    print()
    images_ok = images_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in images_dir.iterdir()
    )
    captions_ok = captions_file.is_file()
    print(f"Flickr8k images present:   {images_ok}")
    print(f"Flickr8k captions present: {captions_ok}")
    print(f"Preprocessing artifacts:   {artifacts_exist()}")
    print(f"Trained model present:     {FINAL_MODEL_PATH.is_file() or BEST_CHECKPOINT_PATH.is_file()}")
    if not images_ok or not captions_ok:
        print(dataset_missing_message())
    elif artifacts_exist():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        captions = json.loads(PROCESSED_CAPTIONS_PATH.read_text(encoding="utf-8"))
        print()
        print_dataset_statistics(metadata, captions)
    else:
        print("Run preprocessing with: python main.py preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionTalk")
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=["status", "preprocess", "extract-features", "train", "generate", "evaluate"],
        help="status (default) lists paths; other commands run that stage only",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="extract-features: first N images")
    parser.add_argument("--image", type=str, default=None, help="generate: image path")
    parser.add_argument("--resume", action="store_true", help="train: resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--beam", action="store_true")
    args = parser.parse_args()

    if args.command == "status":
        _print_menu()
        print()
        _print_status()
        return

    if args.command == "preprocess":
        try:
            run_preprocessing(force=args.force)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        return

    if args.command == "extract-features":
        from src.feature_extraction import extract_features

        try:
            extract_features(limit=args.limit, force=args.force)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        return

    if args.command == "train":
        from src.config import EPOCHS as DEFAULT_EPOCHS
        from src.train import train

        try:
            train(epochs=args.epochs or DEFAULT_EPOCHS, resume=args.resume)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        return

    if args.command == "generate":
        if not args.image:
            print("Error: provide --image path\\to\\image.jpg", file=sys.stderr)
            raise SystemExit(2)
        from src.inference import generate_caption

        try:
            print(generate_caption(args.image, use_beam=args.beam))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        return

    if args.command == "evaluate":
        from src.evaluation import evaluate

        try:
            evaluate()
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()

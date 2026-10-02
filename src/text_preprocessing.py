"""Caption cleaning, sequence tokens, and Flickr8k preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

from src.config import (
    DATA_PROCESSED_DIR,
    END_TOKEN,
    MAX_VOCAB_SIZE,
    METADATA_PATH,
    PROCESSED_CAPTIONS_PATH,
    SEQUENCES_PATH,
    SPLIT_PATH,
    START_TOKEN,
    TOKENIZER_PATH,
    ensure_project_directories,
)
from src.data_loader import load_captions, make_splits
from src.tokenizer_utils import (
    build_vocabulary,
    encode_caption_map,
    max_caption_length,
    save_tokenizer,
)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_caption(text: str) -> str:
    """Lowercase, strip punctuation/unwanted characters, and normalize whitespace."""
    text = text.lower().replace("\n", " ").replace("\r", " ")
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_sequence_tokens(cleaned_text: str) -> str:
    if not cleaned_text:
        return ""
    return f"{START_TOKEN} {cleaned_text} {END_TOKEN}"


def clean_caption_map(raw_captions: dict[str, list[str]]) -> tuple[dict[str, list[str]], int]:
    cleaned: dict[str, list[str]] = {}
    dropped_empty = 0
    for image_name, captions in raw_captions.items():
        kept = []
        for caption in captions:
            processed = add_sequence_tokens(clean_caption(caption))
            if processed:
                kept.append(processed)
            else:
                dropped_empty += 1
        if kept:
            cleaned[image_name] = kept
    return cleaned, dropped_empty


def artifacts_exist() -> bool:
    return all(
        path.is_file()
        for path in (PROCESSED_CAPTIONS_PATH, TOKENIZER_PATH, SPLIT_PATH, METADATA_PATH, SEQUENCES_PATH)
    )


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _captions_for_split(
    caption_map: dict[str, list[str]], image_ids: list[str]
) -> list[str]:
    captions: list[str] = []
    for image_id in image_ids:
        captions.extend(caption_map[image_id])
    return captions


def _example_captions(caption_map: dict[str, list[str]], image_ids: list[str], count: int = 3) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for image_id in sorted(image_ids):
        for caption in caption_map[image_id]:
            examples.append((image_id, caption))
            if len(examples) == count:
                return examples
    return examples


def print_dataset_statistics(metadata: dict, caption_map: dict[str, list[str]] | None = None) -> None:
    print("Dataset statistics")
    print(f"  Images with captions: {metadata['num_images']}")
    print(f"  Total cleaned captions: {metadata['num_captions']}")
    print(f"  Train images: {metadata['num_train_images']}")
    print(f"  Val images:   {metadata['num_val_images']}")
    print(f"  Test images:  {metadata['num_test_images']}")
    print(f"  Train captions: {metadata['num_train_captions']}")
    print(f"  Skipped (no image file): {metadata['skipped_missing_image']}")
    print(f"  Dropped empty after cleaning: {metadata['dropped_empty_captions']}")
    print(f"  Vocabulary size: {metadata['vocab_size']}")
    print(f"  Maximum caption length: {metadata['max_caption_length']}")
    print()
    print("Example cleaned captions")
    examples = metadata.get("example_captions") or []
    if not examples and caption_map is not None:
        examples = [
            {"image": image, "caption": caption}
            for image, caption in _example_captions(caption_map, metadata["splits"]["train"])
        ]
    for item in examples[:3]:
        print(f"  [{item['image']}] {item['caption']}")


def run_preprocessing(
    force: bool = False,
    captions_file: Path | None = None,
    images_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict:
    """Clean captions, split by image, build a train-only vocabulary, and cache artifacts."""
    ensure_project_directories()
    processed_dir = Path(processed_dir or DATA_PROCESSED_DIR)
    if processed_dir != DATA_PROCESSED_DIR:
        captions_out = processed_dir / PROCESSED_CAPTIONS_PATH.name
        sequences_out = processed_dir / SEQUENCES_PATH.name
        tokenizer_out = processed_dir / TOKENIZER_PATH.name
        split_out = processed_dir / SPLIT_PATH.name
        metadata_out = processed_dir / METADATA_PATH.name
    else:
        captions_out = PROCESSED_CAPTIONS_PATH
        sequences_out = SEQUENCES_PATH
        tokenizer_out = TOKENIZER_PATH
        split_out = SPLIT_PATH
        metadata_out = METADATA_PATH

    if not force and all(
        path.is_file()
        for path in (captions_out, sequences_out, tokenizer_out, split_out, metadata_out)
    ):
        metadata = _load_json(metadata_out)
        caption_map = _load_json(captions_out)
        print(f"Loaded cached preprocessing from {processed_dir}")
        print_dataset_statistics(metadata, caption_map)
        return metadata

    raw_captions, skipped_missing_image = load_captions(captions_file, images_dir)
    cleaned, dropped_empty = clean_caption_map(raw_captions)
    splits = make_splits(list(cleaned.keys()))

    train_captions = _captions_for_split(cleaned, splits["train"])
    tokenizer = build_vocabulary(train_captions, max_vocab_size=MAX_VOCAB_SIZE)
    computed_max_length = max_caption_length(train_captions)
    tokenizer["max_caption_length"] = computed_max_length

    sequences = encode_caption_map(cleaned, tokenizer)
    examples = [
        {"image": image, "caption": caption}
        for image, caption in _example_captions(cleaned, splits["train"])
    ]

    metadata = {
        "num_images": len(cleaned),
        "num_captions": sum(len(v) for v in cleaned.values()),
        "num_train_images": len(splits["train"]),
        "num_val_images": len(splits["val"]),
        "num_test_images": len(splits["test"]),
        "num_train_captions": len(train_captions),
        "skipped_missing_image": skipped_missing_image,
        "dropped_empty_captions": dropped_empty,
        "vocab_size": tokenizer["vocab_size"],
        "max_caption_length": computed_max_length,
        "example_captions": examples,
        "splits": {name: len(ids) for name, ids in splits.items()},
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(captions_out, cleaned)
    _dump_json(sequences_out, sequences)
    _dump_json(split_out, splits)
    _dump_json(metadata_out, metadata)
    save_tokenizer(tokenizer, tokenizer_out)

    print(f"Saved processed captions: {captions_out}")
    print(f"Saved sequences:          {sequences_out}")
    print(f"Saved tokenizer:          {tokenizer_out}")
    print(f"Saved splits:             {split_out}")
    print(f"Saved metadata:           {metadata_out}")
    print()
    print_dataset_statistics(metadata, cleaned)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionTalk Flickr8k preprocessing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run preprocessing even if cached artifacts exist.",
    )
    args = parser.parse_args()
    try:
        run_preprocessing(force=args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

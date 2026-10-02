"""Load Flickr8k images and captions from paths in config.

Does not download the dataset. Splits are by image id to avoid caption leakage.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    IMAGE_EXTENSIONS,
    MISSING_IMAGE_RAISE_COUNT,
    MISSING_IMAGE_RAISE_FRACTION,
    RANDOM_SEED,
    TEST_IMAGES_FILE,
    TEST_SPLIT,
    TRAIN_IMAGES_FILE,
    VAL_IMAGES_FILE,
    VALIDATION_SPLIT,
    dataset_missing_message,
    resolve_caption_file,
    resolve_images_dir,
)

CaptionMap = dict[str, list[str]]


def validate_dataset_files(
    captions_file: Path | None = None,
    images_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Raise a clear error if the Flickr8k files are missing."""
    captions_file = Path(captions_file) if captions_file is not None else resolve_caption_file()
    images_dir = Path(images_dir) if images_dir is not None else resolve_images_dir()

    if not captions_file.is_file():
        raise FileNotFoundError(
            f"{dataset_missing_message()}\n"
            f"Caption file not found: {captions_file}\n"
            "Expected either data/raw/Flickr8k_text/Flickr8k.token.txt or "
            "data/raw/Flickr8k/captions.txt. Paths are configured in src/config.py."
        )
    if not images_dir.is_dir() or not any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in images_dir.iterdir()
        if path.exists()
    ):
        raise FileNotFoundError(
            f"{dataset_missing_message()}\n"
            f"Images directory not found or empty: {images_dir}\n"
            "Expected JPEG/PNG files under data/raw/Flickr8k_Dataset/ or "
            "data/raw/Flickr8k/Images/."
        )
    return captions_file, images_dir


def list_image_filenames(images_dir: Path | None = None) -> set[str]:
    images_dir = Path(images_dir) if images_dir is not None else resolve_images_dir()
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset_missing_message()}\nImages directory not found: {images_dir}"
        )
    names: set[str] = set()
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            names.add(path.name)
    if not names:
        raise FileNotFoundError(
            f"No image files ({', '.join(sorted(IMAGE_EXTENSIONS))}) found in {images_dir}."
        )
    return names


def _parse_token_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None
    if "\t" in line:
        left, caption = line.split("\t", 1)
    else:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return None
        left, caption = parts
    image_name = left.split("#", 1)[0].strip()
    caption = caption.strip()
    if not image_name or not caption:
        return None
    return image_name, caption


def load_captions(
    captions_file: Path | None = None,
    images_dir: Path | None = None,
) -> tuple[CaptionMap, int]:
    """Load raw captions keyed by the image filename on disk.

    Returns (image_to_captions, number of caption keys with no matching image).
    """
    captions_file, images_dir = validate_dataset_files(captions_file, images_dir)
    disk_names = list_image_filenames(images_dir)
    name_by_lower = {name.lower(): name for name in disk_names}

    raw: dict[str, list[str]] = defaultdict(list)
    with captions_file.open(encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        handle.seek(0)
        first_lower = first.strip().lower()
        is_csv = first_lower.startswith("image") and "," in first

        if is_csv:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Captions CSV has no header: {captions_file}")
            fields = {name.strip().lower(): name for name in reader.fieldnames if name}
            image_col = fields.get("image")
            caption_col = fields.get("caption")
            if image_col is None or caption_col is None:
                raise ValueError(
                    f"Captions CSV must have 'image' and 'caption' columns. "
                    f"Found: {reader.fieldnames}"
                )
            for row in reader:
                image_name = (row.get(image_col) or "").strip()
                caption = (row.get(caption_col) or "").strip()
                if image_name and caption:
                    raw[image_name].append(caption)
        else:
            for line in handle:
                parsed = _parse_token_line(line)
                if parsed is None:
                    continue
                image_name, caption = parsed
                raw[image_name].append(caption)

    if not raw:
        raise ValueError(f"No captions could be parsed from {captions_file}.")

    matched: CaptionMap = {}
    missing_names: list[str] = []
    for image_name, captions in raw.items():
        actual = name_by_lower.get(image_name.lower())
        if actual is None:
            missing_names.append(image_name)
            continue
        matched[actual] = captions

    skipped_missing_image = len(missing_names)
    if missing_names:
        preview = ", ".join(missing_names[:20])
        extra = f" ... and {len(missing_names) - 20} more" if len(missing_names) > 20 else ""
        print(
            f"Warning: {skipped_missing_image} caption key(s) have no matching image file. "
            f"Examples: {preview}{extra}"
        )
        fraction = skipped_missing_image / max(len(raw), 1)
        if (
            skipped_missing_image >= MISSING_IMAGE_RAISE_COUNT
            or fraction >= MISSING_IMAGE_RAISE_FRACTION
        ):
            raise FileNotFoundError(
                f"{skipped_missing_image} captioned images are missing from {images_dir} "
                f"({fraction:.1%} of caption keys). Fix the dataset layout or paths in "
                "src/config.py before continuing."
            )

    if not matched:
        raise ValueError(
            "No captions matched image files. Check that filenames in the captions "
            f"file correspond to files in {images_dir}."
        )
    return matched, skipped_missing_image


def split_image_ids(
    image_ids: list[str],
    test_split: float = TEST_SPLIT,
    validation_split: float = VALIDATION_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """Split unique image ids. All captions of an image stay in one split."""
    unique_ids = sorted(set(image_ids))
    n = len(unique_ids)
    if n < 3:
        raise ValueError(f"Need at least 3 images to create train/val/test splits, found {n}.")
    if not 0.0 < test_split < 1.0 or not 0.0 < validation_split < 1.0:
        raise ValueError("TEST_SPLIT and VALIDATION_SPLIT must be between 0 and 1.")
    if test_split + validation_split >= 1.0:
        raise ValueError("TEST_SPLIT + VALIDATION_SPLIT must be less than 1.")

    train_val, test = train_test_split(
        unique_ids,
        test_size=test_split,
        random_state=random_seed,
        shuffle=True,
    )
    val_ratio = validation_split / (1.0 - test_split)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=random_seed,
        shuffle=True,
    )
    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def _read_image_id_file(path: Path) -> list[str]:
    names: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            name = line.strip().split()[0] if line.strip() else ""
            if name:
                names.append(name)
    return names


def load_predefined_splits(
    available_images: list[str],
    train_file: Path | None = None,
    val_file: Path | None = None,
    test_file: Path | None = None,
) -> dict[str, list[str]] | None:
    """Load Flickr8k train/dev/test id files if all three exist."""
    train_file = Path(train_file or TRAIN_IMAGES_FILE)
    val_file = Path(val_file or VAL_IMAGES_FILE)
    test_file = Path(test_file or TEST_IMAGES_FILE)
    if not (train_file.is_file() and val_file.is_file() and test_file.is_file()):
        return None

    lookup = {name.lower(): name for name in available_images}
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    mapping = {"train": train_file, "val": val_file, "test": test_file}
    missing_from_disk = 0
    for split_name, path in mapping.items():
        seen: set[str] = set()
        for raw_name in _read_image_id_file(path):
            actual = lookup.get(raw_name.lower())
            if actual is None:
                missing_from_disk += 1
                continue
            if actual not in seen:
                splits[split_name].append(actual)
                seen.add(actual)

    overlap = (
        set(splits["train"]) & set(splits["val"])
        | set(splits["train"]) & set(splits["test"])
        | set(splits["val"]) & set(splits["test"])
    )
    if overlap:
        raise ValueError(
            f"Predefined Flickr8k splits overlap on {len(overlap)} image(s). "
            "Check the train/dev/test id files."
        )
    if any(len(ids) == 0 for ids in splits.values()):
        print("Predefined split files exist but a split is empty after matching images.")
        return None
    if missing_from_disk:
        print(
            f"Warning: {missing_from_disk} id(s) in predefined split files have no matching "
            "image+caption pair and were skipped."
        )
    return {name: sorted(ids) for name, ids in splits.items()}


def make_splits(image_ids: list[str]) -> dict[str, list[str]]:
    """Prefer official Flickr8k splits; otherwise create a seeded image-level split."""
    predefined = load_predefined_splits(image_ids)
    if predefined is not None:
        print("Using predefined Flickr8k train/dev/test image lists.")
        return predefined
    print("Predefined split files not found; creating a reproducible image-level split.")
    return split_image_ids(image_ids)

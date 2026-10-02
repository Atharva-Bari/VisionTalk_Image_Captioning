"""CNN image feature extraction with per-image disk caching.

Run independently of training. Cached vectors are reused; interrupted runs resume
by skipping files that already exist under FEATURES_VECTOR_DIR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.config import (
    CNN_MODEL_NAME,
    CNN_POOLING,
    CNN_WEIGHTS,
    FEATURE_BATCH_SIZE,
    FEATURE_VECTOR_DIM,
    FEATURES_METADATA_PATH,
    FEATURES_VECTOR_DIR,
    IMAGE_SIZE,
    IMAGE_EXTENSIONS,
    ensure_project_directories,
    resolve_images_dir,
)
from src.data_loader import list_image_filenames

EXTRACTOR_PREPROCESS = "tensorflow.keras.applications.vgg16.preprocess_input"


def vector_cache_path(image_name: str, vector_dir: Path | None = None) -> Path:
    directory = Path(vector_dir or FEATURES_VECTOR_DIR)
    return directory / f"{image_name}.npy"


def _load_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)


def expected_extractor_metadata(feature_dim: int) -> dict:
    return {
        "cnn_model_name": CNN_MODEL_NAME,
        "weights": CNN_WEIGHTS,
        "include_top": False,
        "pooling": CNN_POOLING,
        "image_size": list(IMAGE_SIZE),
        "preprocess": EXTRACTOR_PREPROCESS,
        "feature_vector_dim": feature_dim,
    }


def _ensure_metadata_compatible(existing: dict, expected: dict, force: bool) -> None:
    if not existing or force:
        return
    for key in ("cnn_model_name", "weights", "pooling", "preprocess", "feature_vector_dim"):
        if key in existing and existing[key] != expected[key]:
            raise ValueError(
                f"Cached features used {key}={existing[key]!r}, but config expects "
                f"{expected[key]!r}. Re-run with --force to rebuild the cache, or "
                "change src/config.py to match the cached extractor."
            )
    if existing.get("image_size") and existing["image_size"] != expected["image_size"]:
        raise ValueError(
            f"Cached features used image_size={existing['image_size']}, "
            f"but config expects {expected['image_size']}. Re-run with --force to rebuild."
        )


def build_encoder():
    """Frozen VGG16 without the ImageNet classifier; GAP feature vector."""
    from tensorflow.keras.applications.vgg16 import VGG16

    if CNN_MODEL_NAME != "VGG16":
        raise ValueError(
            f"Unsupported CNN_MODEL_NAME={CNN_MODEL_NAME!r}. "
            "Update src/config.py or this module."
        )
    encoder = VGG16(
        weights=CNN_WEIGHTS,
        include_top=False,
        pooling=CNN_POOLING,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    encoder.trainable = False
    return encoder


def _preprocess_batch(images: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.vgg16 import preprocess_input

    return preprocess_input(images)


def load_image_array(image_path: Path) -> np.ndarray:
    from tensorflow.keras.utils import img_to_array, load_img

    image = load_img(image_path, target_size=IMAGE_SIZE, color_mode="rgb")
    return img_to_array(image)


def _is_valid_vector(path: Path, feature_dim: int) -> bool:
    if not path.is_file():
        return False
    try:
        vector = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return vector.shape == (feature_dim,)


def _extract_batch(encoder, arrays: np.ndarray) -> np.ndarray:
    preprocessed = _preprocess_batch(arrays)
    features = encoder.predict(preprocessed, verbose=0)
    return np.asarray(features, dtype=np.float32)


def extract_features(
    images_dir: Path | None = None,
    vector_dir: Path | None = None,
    metadata_path: Path | None = None,
    limit: int | None = None,
    force: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Extract CNN features for images, skipping vectors that are already cached."""
    ensure_project_directories()
    images_dir = Path(images_dir) if images_dir is not None else resolve_images_dir()
    vector_dir = Path(vector_dir or FEATURES_VECTOR_DIR)
    metadata_path = Path(metadata_path or FEATURES_METADATA_PATH)
    batch_size = batch_size or FEATURE_BATCH_SIZE

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            "Place Flickr8k images under data/raw/Flickr8k/Images/ (see src/config.py). "
            "Nothing is downloaded automatically."
        )

    vector_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(list_image_filenames(images_dir))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        names = names[:limit]

    encoder = None
    feature_dim = FEATURE_VECTOR_DIM
    expected_meta = expected_extractor_metadata(feature_dim)
    existing_meta = _load_metadata(metadata_path)
    _ensure_metadata_compatible(existing_meta, expected_meta, force=force)

    skipped = 0
    extracted = 0
    failed: list[str] = list(existing_meta.get("failed_images", [])) if not force else []
    failed_set = set(failed)
    pending: list[str] = []

    def ensure_encoder() -> None:
        nonlocal encoder, feature_dim, expected_meta
        if encoder is not None:
            return
        print(f"Loading pretrained {CNN_MODEL_NAME} (include_top=False, pooling={CNN_POOLING})...")
        encoder = build_encoder()
        feature_dim = int(encoder.output_shape[-1])
        expected_meta = expected_extractor_metadata(feature_dim)

    def flush_pending() -> None:
        nonlocal pending, extracted, skipped
        if not pending:
            return
        ensure_encoder()
        arrays = []
        valid_names = []
        for image_name in pending:
            image_path = images_dir / image_name
            try:
                arrays.append(load_image_array(image_path))
                valid_names.append(image_name)
            except (OSError, ValueError) as exc:
                print(f"  Skipping unreadable image {image_name}: {exc}")
                if image_name not in failed_set:
                    failed.append(image_name)
                    failed_set.add(image_name)
        pending = []
        if not valid_names:
            return
        batch = np.stack(arrays, axis=0)
        vectors = _extract_batch(encoder, batch)
        if vectors.ndim != 2 or vectors.shape[0] != len(valid_names):
            raise RuntimeError(
                f"Unexpected encoder output shape {vectors.shape} for batch of {len(valid_names)}."
            )
        if vectors.shape[1] != feature_dim:
            raise RuntimeError(
                f"Encoder feature size {vectors.shape[1]} does not match expected {feature_dim}."
            )
        for image_name, vector in zip(valid_names, vectors):
            np.save(vector_cache_path(image_name, vector_dir), vector.astype(np.float32, copy=False))
            extracted += 1
            failed_set.discard(image_name)

    total = len(names)
    print(f"Feature extraction: {total} image(s) from {images_dir}")
    print(f"Cache directory: {vector_dir}")
    print(f"Batch size: {batch_size}")

    for index, image_name in enumerate(names, start=1):
        cache_file = vector_cache_path(image_name, vector_dir)
        if not force and _is_valid_vector(cache_file, feature_dim):
            skipped += 1
            failed_set.discard(image_name)
        else:
            pending.append(image_name)
            if len(pending) >= batch_size:
                flush_pending()
        if index % batch_size == 0 or index == total:
            print(
                f"  Progress {index}/{total} "
                f"(extracted {extracted}, cached {skipped}, failed {len(failed_set)})"
            )

    flush_pending()

    cached_now = skipped + extracted
    metadata = {
        **expected_meta,
        "images_dir": str(images_dir),
        "vector_dir": str(vector_dir),
        "num_requested": total,
        "num_extracted_this_run": extracted,
        "num_reused_from_cache": skipped,
        "num_vectors_on_disk": cached_now,
        "failed_images": sorted(failed_set),
        "limit": limit,
    }
    _save_metadata(metadata, metadata_path)
    print(
        f"Done. Extracted {extracted}, reused {skipped}, failed {len(failed_set)}. "
        f"Metadata: {metadata_path}"
    )
    return metadata


def load_features(
    image_names: list[str] | None = None,
    vector_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load cached feature vectors. Does not run the CNN."""
    vector_dir = Path(vector_dir or FEATURES_VECTOR_DIR)
    if image_names is None:
        image_names = [path.name[: -len(".npy")] for path in sorted(vector_dir.glob("*.npy"))]
    features = {}
    missing = []
    for image_name in image_names:
        path = vector_cache_path(image_name, vector_dir)
        if not path.is_file():
            missing.append(image_name)
            continue
        features[image_name] = np.load(path)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} cached feature file(s), e.g. {missing[0]}. "
            "Run: python main.py extract-features"
        )
    return features


def extract_single_image_feature(image_path: Path | str, encoder=None) -> np.ndarray:
    """Extract a feature vector for one image. Does not scan the Flickr8k dataset."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Invalid image format: {path.suffix}. Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    try:
        array = load_image_array(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Corrupted or unreadable image: {path}\n{exc}") from exc
    if encoder is None:
        encoder = build_encoder()
    vectors = _extract_batch(encoder, np.expand_dims(array, axis=0))
    return np.asarray(vectors[0], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionTalk CNN feature extraction")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N image filenames (sorted). Omit to process all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute vectors even when a valid cache file exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Images per CNN batch (default: {FEATURE_BATCH_SIZE}).",
    )
    args = parser.parse_args()
    try:
        extract_features(limit=args.limit, force=args.force, batch_size=args.batch_size)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

"""Extract VGG16's 7x7 spatial feature grid for visual attention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import IMAGE_EXTENSIONS, IMAGE_SIZE
from src.feature_extraction import _preprocess_batch, load_image_array

REGION_COUNT = 49
REGION_DIM = 512


def build_region_encoder():
    from tensorflow.keras import Model
    from tensorflow.keras.applications.vgg16 import VGG16

    base = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    encoder = Model(base.input, base.get_layer("block5_pool").output, name="vgg16_spatial_encoder")
    encoder.trainable = False
    return encoder


def extract_image_regions(image_path: str | Path, encoder=None) -> np.ndarray:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix}")
    if encoder is None:
        encoder = build_region_encoder()
    array = load_image_array(path)
    feature_map = encoder.predict(_preprocess_batch(np.expand_dims(array, 0)), verbose=0)[0]
    if feature_map.shape != (7, 7, REGION_DIM):
        raise ValueError(f"VGG16 spatial feature map has unexpected shape {feature_map.shape}.")
    return np.asarray(feature_map.reshape(REGION_COUNT, REGION_DIM), dtype=np.float32)


def extract_region_cache(
    image_ids: list[str],
    images_dir: Path,
    output_dir: Path,
    *,
    batch_size: int = 16,
    force: bool = False,
) -> dict:
    """Create/reuse one region tensor per image; safe to resume after interruption."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir.parent / "region_extractor_metadata.json"
    expected_metadata = {
        "cnn_model_name": "VGG16",
        "weights": "imagenet",
        "include_top": False,
        "feature_layer": "block5_pool",
        "image_size": list(IMAGE_SIZE),
        "preprocess": "tensorflow.keras.applications.vgg16.preprocess_input",
        "feature_shape": [REGION_COUNT, REGION_DIM],
        "dtype": "float16",
    }
    if metadata_path.is_file() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != expected_metadata:
            raise ValueError(
                "Spatial feature cache metadata does not match the VGG16 attention extractor. "
                "Rebuild with --force-features."
            )
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
    encoder = build_region_encoder()
    extracted = skipped = 0
    pending: list[str] = []

    def flush() -> None:
        nonlocal extracted, pending
        if not pending:
            return
        arrays = np.stack([load_image_array(images_dir / name) for name in pending])
        maps = encoder.predict(_preprocess_batch(arrays), verbose=0)
        for name, fmap in zip(pending, maps):
            path = output_dir / f"{name}.npy"
            np.save(path, np.asarray(fmap, dtype=np.float16).reshape(REGION_COUNT, REGION_DIM))
            extracted += 1
        pending = []

    for i, name in enumerate(image_ids, 1):
        path = output_dir / f"{name}.npy"
        if path.is_file() and not force:
            try:
                if np.load(path, mmap_mode="r").shape == (REGION_COUNT, REGION_DIM):
                    skipped += 1
                    continue
            except (OSError, ValueError):
                pass
        pending.append(name)
        if len(pending) >= batch_size:
            flush()
        if i % 200 == 0 or i == len(image_ids):
            print(f"Region features {i}/{len(image_ids)} (new {extracted}, cached {skipped})")
    flush()
    return {"images": len(image_ids), "extracted": extracted, "cached": skipped,
            "shape": [REGION_COUNT, REGION_DIM], "dtype": "float16",
            "metadata": str(metadata_path)}

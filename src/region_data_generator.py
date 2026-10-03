"""Teacher-forcing batches for cached spatial image features."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import BATCH_SIZE, RANDOM_SEED, REGION_FEATURES_DIR
from src.region_feature_extraction import REGION_COUNT, REGION_DIM
from src.tokenizer_utils import pad_sequences


def make_region_tf_dataset(
    image_ids: list[str], sequences_map: dict[str, list[list[int]]], max_length: int,
    vocab_size: int, *, batch_size: int = BATCH_SIZE,
    feature_dir: Path = REGION_FEATURES_DIR, shuffle: bool = True, seed: int = RANDOM_SEED,
):
    import tensorflow as tf

    usable = [name for name in image_ids if name in sequences_map and
              (feature_dir / f"{name}.npy").is_file()]
    missing = len(image_ids) - len(usable)
    if not usable:
        raise FileNotFoundError(f"No region features found in {feature_dir}. Run: python -m src.train_attention --extract-only")
    if missing:
        print(f"Warning: {missing} images have no region feature cache and will be skipped.")
    caption_count = sum(1 for image_id in usable for seq in sequences_map[image_id] if len(seq) >= 2)
    steps = max(1, (caption_count + batch_size - 1) // batch_size)

    def batches():
        rng = np.random.default_rng(seed)
        while True:
            order = list(usable)
            if shuffle:
                rng.shuffle(order)
            xb: list[np.ndarray] = []
            sb: list[list[int]] = []
            yb: list[list[int]] = []
            for image_id in order:
                feature = np.load(feature_dir / f"{image_id}.npy").astype(np.float32)
                if feature.shape != (REGION_COUNT, REGION_DIM):
                    raise ValueError(f"Bad region feature shape for {image_id}: {feature.shape}")
                for sequence in sequences_map[image_id]:
                    if len(sequence) < 2:
                        continue
                    xb.append(feature)
                    sb.append(sequence[:-1])
                    yb.append(sequence[1:])
                    if len(yb) == batch_size:
                        tokens = pad_sequences(sb, max_length)
                        targets = pad_sequences(yb, max_length)
                        weights = (targets != 0).astype(np.float32)
                        yield ((np.stack(xb), tokens), targets, weights)
                        xb, sb, yb = [], [], []
            if yb:
                tokens = pad_sequences(sb, max_length)
                targets = pad_sequences(yb, max_length)
                weights = (targets != 0).astype(np.float32)
                yield ((np.stack(xb), tokens), targets, weights)

    dataset = tf.data.Dataset.from_generator(
        batches,
        output_signature=((tf.TensorSpec((None, REGION_COUNT, REGION_DIM), tf.float32),
                           tf.TensorSpec((None, max_length), tf.int32)),
                          tf.TensorSpec((None, max_length), tf.int32),
                          tf.TensorSpec((None, max_length), tf.float32)),
    ).prefetch(tf.data.AUTOTUNE)
    return dataset, steps, usable

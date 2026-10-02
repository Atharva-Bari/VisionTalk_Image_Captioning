"""Teacher-forcing batch pipeline. Yields batches; does not materialize every sample."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.config import BATCH_SIZE, FEATURE_VECTOR_DIM, FEATURES_VECTOR_DIR, RANDOM_SEED
from src.feature_extraction import vector_cache_path
from src.tokenizer_utils import pad_sequences


def count_teacher_forcing_samples(sequences_map: dict[str, list[list[int]]], image_ids: list[str]) -> int:
    total = 0
    for image_id in image_ids:
        for sequence in sequences_map.get(image_id, []):
            if len(sequence) >= 2:
                total += len(sequence) - 1
    return total


def _images_with_features(
    image_ids: list[str],
    sequences_map: dict[str, list[list[int]]],
    vector_dir: Path,
) -> list[str]:
    usable = []
    missing = []
    for image_id in image_ids:
        if image_id not in sequences_map:
            continue
        if vector_cache_path(image_id, vector_dir).is_file():
            usable.append(image_id)
        else:
            missing.append(image_id)
    if not usable:
        raise FileNotFoundError(
            f"No cached CNN features found under {vector_dir}. "
            "Run: python -m src.feature_extraction"
        )
    if missing:
        print(
            f"Warning: skipping {len(missing)} image(s) without cached features "
            f"(example: {missing[0]})."
        )
    return usable


def iter_teacher_forcing_batches(
    image_ids: list[str],
    sequences_map: dict[str, list[list[int]]],
    vector_dir: Path | None = None,
    batch_size: int = BATCH_SIZE,
    max_length: int = 1,
    feature_dim: int = FEATURE_VECTOR_DIM,
    shuffle: bool = True,
    seed: int = RANDOM_SEED,
) -> Iterator[tuple[tuple[np.ndarray, np.ndarray], np.ndarray]]:
    """Yield ((image_batch, sequence_batch), next_word_ids) indefinitely."""
    vector_dir = Path(vector_dir or FEATURES_VECTOR_DIR)
    usable = _images_with_features(image_ids, sequences_map, vector_dir)
    rng = np.random.default_rng(seed)
    while True:
        order = list(usable)
        if shuffle:
            rng.shuffle(order)
        image_batch: list[np.ndarray] = []
        seq_batch: list[list[int]] = []
        y_batch: list[int] = []
        for image_id in order:
            feature = np.load(vector_cache_path(image_id, vector_dir)).astype(np.float32, copy=False)
            if feature.shape != (feature_dim,):
                raise ValueError(
                    f"Feature for {image_id} has shape {feature.shape}, expected ({feature_dim},)."
                )
            for sequence in sequences_map[image_id]:
                if len(sequence) < 2:
                    continue
                for t in range(1, len(sequence)):
                    seq_batch.append(sequence[:t])
                    image_batch.append(feature)
                    y_batch.append(int(sequence[t]))
                    if len(y_batch) == batch_size:
                        yield (
                            (
                                np.stack(image_batch, axis=0),
                                pad_sequences(seq_batch, max_length),
                            ),
                            np.asarray(y_batch, dtype=np.int32),
                        )
                        image_batch, seq_batch, y_batch = [], [], []
        if y_batch:
            yield (
                (
                    np.stack(image_batch, axis=0),
                    pad_sequences(seq_batch, max_length),
                ),
                np.asarray(y_batch, dtype=np.int32),
            )


def make_tf_dataset(
    image_ids: list[str],
    sequences_map: dict[str, list[list[int]]],
    max_length: int,
    vocab_size: int,
    vector_dir: Path | None = None,
    batch_size: int = BATCH_SIZE,
    feature_dim: int = FEATURE_VECTOR_DIM,
    shuffle: bool = True,
    seed: int = RANDOM_SEED,
):
    """tf.data wrapper around the teacher-forcing generator."""
    import tensorflow as tf

    vector_dir = Path(vector_dir or FEATURES_VECTOR_DIR)
    usable = _images_with_features(image_ids, sequences_map, vector_dir)
    steps = max(1, (count_teacher_forcing_samples(sequences_map, usable) + batch_size - 1) // batch_size)

    output_signature = (
        (
            tf.TensorSpec(shape=(None, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, max_length), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: iter_teacher_forcing_batches(
            usable,
            sequences_map,
            vector_dir=vector_dir,
            batch_size=batch_size,
            max_length=max_length,
            feature_dim=feature_dim,
            shuffle=shuffle,
            seed=seed,
        ),
        output_signature=output_signature,
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, steps, usable

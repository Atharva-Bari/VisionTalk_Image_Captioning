"""Shared helpers: seeds, JSON I/O, caption token stripping, logging."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np

from src.config import END_TOKEN, RANDOM_SEED, START_TOKEN


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("visiontalk")


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Make split / numpy / TensorFlow draws reproducible where possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def strip_sequence_tokens(text: str) -> str:
    """Remove startseq / endseq from a displayed or BLEU caption."""
    tokens = [token for token in text.split() if token and token not in {START_TOKEN, END_TOKEN}]
    return " ".join(tokens)


def caption_tokens_for_bleu(text: str) -> list[str]:
    return strip_sequence_tokens(text).split()

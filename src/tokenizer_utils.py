"""Vocabulary, integer sequences, padding, and tokenizer persistence."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import (
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    TOKENIZER_PATH,
    UNK_TOKEN,
)
from src.utils import strip_sequence_tokens


def build_vocabulary(train_captions: list[str], max_vocab_size: int | None = None) -> dict:
    """Build word/id maps from training captions only (includes startseq/endseq)."""
    counts: Counter[str] = Counter()
    for caption in train_captions:
        counts.update(token for token in caption.split() if token)

    reserved = {PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN}
    ordered = [word for word, _ in counts.most_common() if word not in reserved]
    if max_vocab_size is not None:
        keep = max(max_vocab_size - len(reserved), 0)
        ordered = ordered[:keep]

    word_to_index = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    next_index = 2
    for word in (START_TOKEN, END_TOKEN):
        word_to_index[word] = next_index
        next_index += 1
    for word in ordered:
        if word in word_to_index:
            continue
        word_to_index[word] = next_index
        next_index += 1

    index_to_word = {index: word for word, index in word_to_index.items()}
    tokenizer = {
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "vocab_size": len(word_to_index),
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "start_token": START_TOKEN,
        "end_token": END_TOKEN,
    }
    return tokenizer


def max_caption_length(captions: list[str]) -> int:
    if not captions:
        raise ValueError("Cannot compute max caption length from an empty caption list.")
    return max(len(caption.split()) for caption in captions)


def texts_to_sequences(captions: list[str], tokenizer: dict) -> list[list[int]]:
    word_to_index: dict[str, int] = tokenizer["word_to_index"]
    unk_id = word_to_index[tokenizer["unk_token"]]
    sequences = []
    for caption in captions:
        sequences.append([word_to_index.get(token, unk_id) for token in caption.split() if token])
    return sequences


def encode_caption_map(caption_map: dict[str, list[str]], tokenizer: dict) -> dict[str, list[list[int]]]:
    encoded = {}
    for image_name, captions in caption_map.items():
        encoded[image_name] = texts_to_sequences(captions, tokenizer)
    return encoded


def pad_sequences(sequences: list[list[int]], max_length: int, pad_value: int = 0) -> np.ndarray:
    padded = np.full((len(sequences), max_length), pad_value, dtype=np.int32)
    for i, sequence in enumerate(sequences):
        truncated = sequence[-max_length:]
        start = max_length - len(truncated)
        padded[i, start:] = truncated
    return padded


def save_tokenizer(tokenizer: dict, path: Path | None = None) -> Path:
    path = Path(path or TOKENIZER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(tokenizer, handle, protocol=4)
    return path


def load_tokenizer(path: Path | None = None) -> dict:
    path = Path(path or TOKENIZER_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {path}. Run preprocessing first: python main.py preprocess"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def index_lookup(tokenizer: dict) -> dict[int, str]:
    raw = tokenizer["index_to_word"]
    return {int(index): word for index, word in raw.items()}


def sequence_to_text(sequence: list[int], tokenizer: dict, strip_special: bool = True) -> str:
    lookup = index_lookup(tokenizer)
    words = [lookup.get(int(index), tokenizer["unk_token"]) for index in sequence if int(index) != 0]
    text = " ".join(words)
    return strip_sequence_tokens(text) if strip_special else text

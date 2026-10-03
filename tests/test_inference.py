"""Greedy decoding and startseq/endseq handling without Flickr8k."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import greedy_decode
from src.tokenizer_utils import build_vocabulary
from src.utils import strip_sequence_tokens


class _FakeModel:
    def __init__(self, token_order: list[int]) -> None:
        self._token_order = token_order
        self._step = 0
        self.input_shape = [(None, 8), (None, 6)]

    def predict(self, _inputs, verbose=0):
        vocab = max(self._token_order) + 1
        probs = np.zeros((1, vocab), dtype=np.float32)
        token = self._token_order[min(self._step, len(self._token_order) - 1)]
        self._step += 1
        probs[0, token] = 1.0
        return probs


def test_greedy_stops_at_endseq() -> None:
    tokenizer = build_vocabulary(["startseq a dog runs endseq"])
    w2i = tokenizer["word_to_index"]
    fake = _FakeModel([w2i["a"], w2i["dog"], w2i["endseq"], w2i["runs"]])
    caption = greedy_decode(fake, np.zeros((8,), dtype=np.float32), tokenizer, max_length=6)
    assert "startseq" not in caption
    assert "endseq" not in caption
    assert caption == "a dog"
    assert strip_sequence_tokens("startseq a dog endseq") == "a dog"


def test_greedy_respects_max_length() -> None:
    tokenizer = build_vocabulary(["startseq a dog runs endseq"])
    w2i = tokenizer["word_to_index"]
    fake = _FakeModel([w2i["a"], w2i["dog"], w2i["runs"], w2i["a"], w2i["dog"]])
    caption = greedy_decode(fake, np.zeros((8,), dtype=np.float32), tokenizer, max_length=4)
    tokens = caption.split()
    assert len(tokens) <= 3


def main() -> None:
    test_greedy_stops_at_endseq()
    test_greedy_respects_max_length()
    print("test_inference passed.")


if __name__ == "__main__":
    main()

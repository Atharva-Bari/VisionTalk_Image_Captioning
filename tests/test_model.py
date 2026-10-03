"""Model construction, dummy forward pass, and one-batch generator."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_generator import count_teacher_forcing_samples, iter_teacher_forcing_batches
from src.model import build_caption_model, dummy_forward_pass
from src.tokenizer_utils import build_vocabulary, encode_caption_map


def test_model_output_shape() -> None:
    vocab_size = 24
    max_length = 7
    feature_dim = 16
    model = build_caption_model(
        vocab_size=vocab_size,
        max_length=max_length,
        feature_dim=feature_dim,
        embedding_dim=8,
        lstm_units=8,
        dense_units=8,
        dropout=0.0,
    )
    assert int(model.inputs[0].shape[-1]) == feature_dim
    assert int(model.inputs[1].shape[1]) == max_length
    assert int(model.outputs[0].shape[-1]) == vocab_size
    preds = dummy_forward_pass(model, batch_size=3)
    assert preds.shape == (3, vocab_size)


def test_one_batch_generator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vector_dir = Path(tmp)
        captions = {
            "a.jpg": ["startseq a dog runs endseq"],
            "b.jpg": ["startseq a cat sits endseq"],
            "c.jpg": ["startseq a bird flies endseq"],
        }
        tokenizer = build_vocabulary([cap for caps in captions.values() for cap in caps])
        sequences = encode_caption_map(captions, tokenizer)
        feature_dim = 8
        for name in captions:
            np.save(vector_dir / f"{name}.npy", np.ones((feature_dim,), dtype=np.float32))
        max_length = tokenizer["max_caption_length"] if "max_caption_length" in tokenizer else 6
        max_length = max(len(seq) for seqs in sequences.values() for seq in seqs)
        sample_count = count_teacher_forcing_samples(sequences, list(captions))
        assert sample_count > 0
        batch, _ = next(
            iter_teacher_forcing_batches(
                list(captions),
                sequences,
                vector_dir=vector_dir,
                batch_size=4,
                max_length=max_length,
                feature_dim=feature_dim,
                shuffle=False,
            )
        )
        images, tokens = batch
        assert images.shape == (4, feature_dim)
        assert tokens.shape == (4, max_length)


def main() -> None:
    test_model_output_shape()
    test_one_batch_generator()
    print("test_model passed.")


if __name__ == "__main__":
    main()

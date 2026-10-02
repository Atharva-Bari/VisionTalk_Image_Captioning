"""Greedy (and optional beam-search) caption generation.

Never trains. Never scans the full Flickr8k dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.config import (
    BEAM_SIZE,
    BEST_CHECKPOINT_PATH,
    END_TOKEN,
    FEATURE_VECTOR_DIM,
    FINAL_MODEL_PATH,
    IMAGE_EXTENSIONS,
    START_TOKEN,
    TOKENIZER_PATH,
)
from src.feature_extraction import build_encoder, build_encoder_for_dim, extract_single_image_feature
from src.tokenizer_utils import load_tokenizer, pad_sequences, sequence_to_text
from src.utils import strip_sequence_tokens


def resolve_model_path(model_path: Path | None = None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trained model not found: {path}")
        return path
    for candidate in (FINAL_MODEL_PATH, BEST_CHECKPOINT_PATH):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Trained model not found. Train first:\n"
        "  python -m src.train\n"
        f"Expected {FINAL_MODEL_PATH} or {BEST_CHECKPOINT_PATH}."
    )


def load_caption_model(model_path: Path | None = None):
    from tensorflow.keras.models import load_model

    path = resolve_model_path(model_path)
    return load_model(path), path


def _special_ids(tokenizer: dict) -> tuple[int, int]:
    word_to_index = tokenizer["word_to_index"]
    start_id = word_to_index[tokenizer.get("start_token", START_TOKEN)]
    end_id = word_to_index[tokenizer.get("end_token", END_TOKEN)]
    return start_id, end_id


def _suppress_repetitions(probs, seq, block_size: int = 3, penalty: float = 1e-9) -> np.ndarray:
    """Suppress tokens that would create obvious n-gram repetitions in greedy decode.

    Also zeros out tokens that were already output in the last `block_size` positions,
    which prevents the "behind behind behind..." infinite loop common to
    under-trained or ReLU-collapsed LSTM captioners.
    """
    probs = np.asarray(probs, dtype=np.float64)
    recent = seq[-block_size:] if len(seq) >= block_size else seq
    for tok in recent:
        if tok >= len(probs):
            continue
        probs[tok] *= penalty
    return probs


def _break_repetition_loop(seq: list[int], min_run: int = 3) -> bool:
    """Return True if the tail of `seq` repeats the same token `min_run` times in a row."""
    if len(seq) < min_run + 1:
        return False
    tail = seq[-min_run:]
    return all(t == tail[0] and tail[0] != 0 for t in tail)


def greedy_decode(
    model,
    image_feature: np.ndarray,
    tokenizer: dict,
    max_length: int | None = None,
) -> str:
    """Predict the next word until endseq, max_length, or a repetition loop is detected.

    Adds a lightweight repetition suppressor so the same word can't dominate the
    output.  If the model enters a tight repetition loop (e.g. "behind behind behind")
    we stop early rather than keep dumping the same token.
    """
    start_id, end_id = _special_ids(tokenizer)
    max_length = int(max_length or tokenizer.get("max_caption_length") or 20)
    feature = np.asarray(image_feature, dtype=np.float32).reshape(1, -1)
    token_ids = [start_id]
    for _ in range(max_length):
        padded = pad_sequences([token_ids], max_length)
        raw_probs = model.predict([feature, padded], verbose=0)[0]
        probs = _suppress_repetitions(raw_probs, token_ids[1:], block_size=4, penalty=1e-10)
        next_id = int(np.argmax(probs))
        if next_id == end_id or next_id == 0:
            break
        token_ids.append(next_id)
        if _break_repetition_loop(token_ids[1:], min_run=3):
            break
        if len(token_ids) >= max_length:
            break
    return sequence_to_text(token_ids, tokenizer, strip_special=True)


def beam_search_decode(
    model,
    image_feature: np.ndarray,
    tokenizer: dict,
    beam_size: int = BEAM_SIZE,
    max_length: int | None = None,
) -> str:
    """Beam search with repetition penalty and length normalization.

    For each candidate beam we down-rank tokens that would cause an immediate
    repetition of the previous word (prevents the "behind behind behind" degenerate
    output) and apply a hard break for any beam that enters a 3-token repetition
    loop.  Length normalization is already applied in the final sort, so short
    valid captions can beat padded or degenerate long ones.
    """
    start_id, end_id = _special_ids(tokenizer)
    max_length = int(max_length or tokenizer.get("max_caption_length") or 20)
    feature = np.asarray(image_feature, dtype=np.float32).reshape(1, -1)
    beams: list[tuple[float, list[int], bool]] = [(0.0, [start_id], False)]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_length):
        expanded: list[tuple[float, list[int], bool]] = []
        for score, seq, done in beams:
            if done:
                completed.append((score, seq))
                continue
            if _break_repetition_loop(seq[1:], min_run=3):
                completed.append((score, seq))
                continue
            padded = pad_sequences([seq], max_length)
            raw_probs = model.predict([feature, padded], verbose=0)[0]
            probs = _suppress_repetitions(raw_probs, seq[1:], block_size=3, penalty=1e-8)
            top_ids = np.argsort(probs)[-beam_size:]
            for token_id in top_ids:
                token_id = int(token_id)
                logp = float(np.log(max(probs[token_id], 1e-12)))
                new_seq = seq + [token_id]
                finished = (token_id == end_id or token_id == 0 or
                            _break_repetition_loop(new_seq[1:], min_run=3))
                expanded.append((score + logp, new_seq, finished))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[0] / max(len(item[1]), 1), reverse=True)
        beams = expanded[:beam_size]
        if all(done for _, _, done in beams):
            completed.extend((score, seq) for score, seq, _ in beams)
            break

    if not completed:
        completed = [(score, seq) for score, seq, _ in beams]
    completed.sort(key=lambda item: item[0] / max(len(item[1]), 1), reverse=True)
    return sequence_to_text(completed[0][1], tokenizer, strip_special=True)


def generate_caption(
    image_path: str | Path,
    model=None,
    tokenizer: dict | None = None,
    encoder=None,
    use_beam: bool = False,
    beam_size: int = BEAM_SIZE,
) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Invalid image format {path.suffix}. Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    if tokenizer is None:
        tokenizer = load_tokenizer(TOKENIZER_PATH)
    if model is None:
        model, _ = load_caption_model()
    expected_dim = int(model.inputs[0].shape[-1] or FEATURE_VECTOR_DIM)
    if encoder is None:
        encoder = build_encoder_for_dim(expected_dim)
    feature = extract_single_image_feature(path, encoder=encoder)
    if feature.shape[-1] != expected_dim:
        rebuilt_encoder = build_encoder_for_dim(expected_dim)
        feature = extract_single_image_feature(path, encoder=rebuilt_encoder)
        if feature.shape[-1] != expected_dim:
            raise ValueError(
                f"CNN feature size {feature.shape[-1]} does not match model input {expected_dim}. "
                "Retrain after changing CNN_MODEL_NAME, or extract features with the same encoder."
            )
        encoder = rebuilt_encoder
    if use_beam:
        caption = beam_search_decode(model, feature, tokenizer, beam_size=beam_size)
    else:
        caption = greedy_decode(model, feature, tokenizer)
    return strip_sequence_tokens(caption)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a caption for one image")
    parser.add_argument("--image", required=True, help="Path to a JPEG or PNG image.")
    parser.add_argument("--beam", action="store_true", help="Use beam search instead of greedy decoding.")
    parser.add_argument("--beam-size", type=int, default=BEAM_SIZE)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    try:
        caption = generate_caption(
            args.image,
            model=load_caption_model(Path(args.model) if args.model else None)[0],
            use_beam=args.beam,
            beam_size=args.beam_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(caption)


if __name__ == "__main__":
    main()

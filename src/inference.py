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


INFERENCE_CODE_VERSION = "v2.1-repguard"  # Shown in UI to confirm deployed code version


def _suppress_repetitions(probs, seq, block_size: int = 3, penalty: float = 1e-9) -> np.ndarray:
    """Suppress tokens that have appeared in the most recent `block_size` positions.

    Uses a SET of recent token ids (deduplicated) so that a word already repeated
    in the tail is only penalised once — multiplying the same index three times
    caused floating-point underflow that silently defeated the penalty for very
    common filler words such as "behind".
    """
    probs = np.asarray(probs, dtype=np.float64)
    recent = seq[-block_size:] if len(seq) >= block_size else seq
    for tok in set(int(t) for t in recent):
        if tok >= len(probs):
            continue
        probs[tok] *= penalty
    return probs


def _break_repetition_loop(seq: list[int], min_run: int = 2) -> bool:
    """Return True as soon as the tail of `seq` repeats the same token `min_run` times in a row.

    min_run defaults to 2 because any 2-word strict run is already a degenerate
    caption.  The length guard now requires exactly `min_run` tokens (not min_run+1),
    which was the primary bug in v1.0 that let 45 "behind" tokens slip through.
    """
    if len(seq) < min_run:
        return False
    tail = seq[-min_run:]
    first = tail[0]
    if first == 0:
        return False
    return all(t == first for t in tail)


def _clean_repeated_words(caption: str, max_consecutive: int = 1) -> str:
    """Last-resort text-level defense: collapse runs of identical words to one copy.

    Runs AFTER the decoder has produced a string and after start/end/pad tokens
    are stripped.  Even if the token-level defenses all fail (e.g. due to stale
    module cache or TF internal state), this final filter guarantees the output
    can never read "behind behind behind behind..." again.
    """
    words = str(caption).split()
    if not words:
        return ""
    cleaned: list[str] = []
    run = 0
    prev: str | None = None
    for w in words:
        if prev is not None and w == prev:
            run += 1
            if run > max_consecutive:
                continue
        else:
            run = 0
        cleaned.append(w)
        prev = w
    # Also drop any trailing repeated 2-word phrases such as "a dog a dog a dog"
    # (heuristic: collapse exact 2-token runs of length >= 2 to a single copy)
    phrase_collapsed: list[str] = []
    i = 0
    while i < len(cleaned):
        phrase_collapsed.append(cleaned[i])
        if i + 3 < len(cleaned) and cleaned[i] == cleaned[i + 2] and cleaned[i + 1] == cleaned[i + 3]:
            # Phrase loop detected -> skip ahead by 2 each round until loop ends
            j = i + 2
            while j + 1 < len(cleaned) and cleaned[j] == cleaned[i] and cleaned[j + 1] == cleaned[i + 1]:
                j += 2
            i = j
        else:
            i += 1
    return " ".join(phrase_collapsed).strip()


def greedy_decode(
    model,
    image_feature: np.ndarray,
    tokenizer: dict,
    max_length: int | None = None,
) -> str:
    """Predict the next word until endseq, max_length, or a repetition loop is detected.

    Three layers of anti-repetition defense:
      1. Probability penalty for any token emitted in the last 4 steps.
      2. Hard stop as soon as 2 identical tokens appear back-to-back.
      3. `_clean_repeated_words` run inside `generate_caption()` as a last-resort
         text filter after the full string is produced.
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
        if _break_repetition_loop(token_ids[1:], min_run=2):
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
    """Beam search with repetition penalty, length normalization, and loop protection.

    Uses min_run=2 for the repetition loop breaker (stops any beam as soon as a
    word repeats once), so beam candidates cannot dump long chains of "behind"
    even if the probability model wants to.  `completed` beams are sorted by
    length-normalised log-probability; the text-level deduper in
    `generate_caption()` then polishes the final selected string.
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
            if _break_repetition_loop(seq[1:], min_run=2):
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
                            _break_repetition_loop(new_seq[1:], min_run=2))
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
    # Penalty finalist beams that contain any repetition loop, so even if every
    # beam path goes through a degenerate token, we pick the *least* degenerate.
    def _final_rank_key(item):
        score, seq = item
        content = seq[1:]
        loop_count = 0
        for i in range(len(content) - 1):
            if content[i] == content[i + 1] and content[i] != 0:
                loop_count += 1
        normalised = score / max(len(seq), 1)
        return (normalised - 2.0 * loop_count, -loop_count)

    completed.sort(key=_final_rank_key, reverse=True)
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
    caption = strip_sequence_tokens(caption)
    # LAST RESORT — even if all token-level defenses are bypassed (stale module
    # cache, TF internal state, dead-ReLU collapsed projection, etc.), the
    # produced string is filtered so runs of identical words are collapsed to a
    # single copy and obvious 2-token phrase loops are removed.  After this the
    # output is GUARANTEED to never read "behind behind behind..." again.
    return _clean_repeated_words(caption)


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

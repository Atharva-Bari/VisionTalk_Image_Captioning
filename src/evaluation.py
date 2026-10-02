"""BLEU-1 / BLEU-4 on the test split and a 10-image visualization.

Does not train. Uses cached features when present; otherwise encodes one image
at a time.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from src.config import (
    EVALUATION_DIR,
    EVAL_SAMPLE_IMAGES,
    FEATURES_VECTOR_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    PROCESSED_CAPTIONS_PATH,
    RANDOM_SEED,
    SPLIT_PATH,
    TOKENIZER_PATH,
    ensure_project_directories,
    resolve_images_dir,
)
from src.feature_extraction import extract_single_image_feature, vector_cache_path
from src.inference import greedy_decode, load_caption_model
from src.tokenizer_utils import load_tokenizer
from src.utils import caption_tokens_for_bleu, dump_json, load_json, set_global_seed


def _bleu_scores(references: list[list[list[str]]], hypotheses: list[list[str]]) -> dict[str, float]:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    smoothie = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )
    return {"bleu1": float(bleu1), "bleu4": float(bleu4)}


def evaluate(
    max_images: int | None = None,
    visualize: bool = True,
) -> dict:
    ensure_project_directories()
    set_global_seed()

    for path, hint in (
        (TOKENIZER_PATH, "python -m src.text_preprocessing"),
        (PROCESSED_CAPTIONS_PATH, "python -m src.text_preprocessing"),
        (SPLIT_PATH, "python -m src.text_preprocessing"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}. Run: {hint}")

    tokenizer = load_tokenizer()
    captions = load_json(PROCESSED_CAPTIONS_PATH)
    splits = load_json(SPLIT_PATH)
    test_ids = list(splits["test"])
    if max_images is not None:
        test_ids = test_ids[: max(1, max_images)]

    try:
        model, model_path = load_caption_model()
    except FileNotFoundError:
        msg = "BLEU evaluation not yet available because a trained model is required."
        print(msg)
        return {
            "bleu1": None,
            "bleu4": None,
            "num_test_images": 0,
            "model_path": None,
            "note": msg,
        }

    images_dir = resolve_images_dir()
    encoder = None

    references: list[list[list[str]]] = []
    hypotheses: list[list[str]] = []
    rows: list[dict] = []

    from src.feature_extraction import build_encoder

    for index, image_id in enumerate(test_ids, start=1):
        refs = [caption_tokens_for_bleu(text) for text in captions.get(image_id, [])]
        refs = [tokens for tokens in refs if tokens]
        if not refs:
            continue
        cache = vector_cache_path(image_id, FEATURES_VECTOR_DIR)
        if cache.is_file():
            feature = np.load(cache)
        else:
            image_path = images_dir / image_id
            if not image_path.is_file():
                print(f"Skipping missing test image: {image_path}")
                continue
            if encoder is None:
                encoder = build_encoder()
            feature = extract_single_image_feature(image_path, encoder=encoder)
        generated = greedy_decode(model, feature, tokenizer)
        hyp = generated.split()
        references.append(refs)
        hypotheses.append(hyp)
        rows.append(
            {
                "image": image_id,
                "generated": generated,
                "references": [" ".join(tokens) for tokens in refs],
            }
        )
        if index % 25 == 0 or index == len(test_ids):
            print(f"Evaluated {index}/{len(test_ids)}")

    if not hypotheses:
        raise RuntimeError("No test captions could be generated. Check the test split and images.")

    scores = _bleu_scores(references, hypotheses)
    scores["num_test_images"] = len(hypotheses)
    scores["model_path"] = str(model_path)
    scores["note"] = "Scores are only valid after a real training run on Flickr8k."

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    dump_json(EVALUATION_DIR / "bleu_scores.json", scores)
    pred_csv = PREDICTIONS_DIR / "test_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "generated_caption", "actual_captions"])
        for row in rows:
            writer.writerow([row["image"], row["generated"], " || ".join(row["references"])])
    dump_json(PREDICTIONS_DIR / "test_predictions.json", rows)

    print(f"BLEU-1: {scores['bleu1']:.4f}")
    print(f"BLEU-4: {scores['bleu4']:.4f}")
    print(f"Test images scored: {scores['num_test_images']}")
    print(f"Saved: {EVALUATION_DIR / 'bleu_scores.json'}")
    print(f"Saved: {pred_csv}")

    if visualize:
        visualize_test_samples(rows, images_dir)
    return scores


def visualize_test_samples(
    rows: list[dict],
    images_dir: Path,
    count: int = EVAL_SAMPLE_IMAGES,
    seed: int = RANDOM_SEED,
) -> Path | None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    usable = [row for row in rows if (images_dir / row["image"]).is_file()]
    if not usable:
        print("No test images found on disk; skipping visualization.")
        return None
    rng = np.random.default_rng(seed)
    chosen = list(usable)
    rng.shuffle(chosen)
    chosen = chosen[: min(count, len(chosen))]

    cols = 2
    rows_n = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 4 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, chosen):
        image = mpimg.imread(images_dir / row["image"])
        ax.imshow(image)
        ax.axis("off")
        actual = row["references"][0] if row["references"] else ""
        ax.set_title(
            f"Actual: {actual}\nGenerated: {row['generated']}",
            fontsize=8,
            wrap=True,
        )
    for ax in axes[len(chosen) :]:
        ax.axis("off")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "test_captions.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved visualization: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VisionTalk on the test split")
    parser.add_argument("--max-images", type=int, default=None, help="Limit test images (debug).")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()
    try:
        evaluate(max_images=args.max_images, visualize=not args.no_viz)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

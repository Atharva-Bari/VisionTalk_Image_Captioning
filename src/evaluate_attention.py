"""Qualitative held-out checks for the spatial-attention candidate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.config import PREDICTIONS_DIR, PROCESSED_CAPTIONS_PATH, SPLIT_PATH, TOKENIZER_PATH
from src.evaluation import _bleu_scores
from src.inference import beam_search_decode
from src.region_data_generator import REGION_FEATURES_DIR
from src.tokenizer_utils import load_tokenizer
from src.utils import caption_tokens_for_bleu, dump_json, load_json
from src.train_attention import ATTENTION_FINAL_PATH


def diagnostic_images(test_ids: list[str], captions: dict, count: int) -> list[str]:
    groups = [
        ("mountain climbing", ("climb", "climber", "mountaineer")),
        ("dog", ("dog", "puppy")),
        ("street", ("street", "traffic", "city", "sidewalk")),
        ("bicycle", ("bicycle", "bike", "cycling")),
        ("people", ("people", "group", "children", "woman", "man")),
    ]
    chosen: list[str] = []
    for group, terms in groups:
        for image_id in test_ids:
            text = " ".join(captions.get(image_id, [])).lower()
            if group == "mountain climbing":
                matches = any(term in text for term in terms) and any(
                    scene in text for scene in ("mountain", "rock", "cliff", "glacier", "ice")
                )
            else:
                matches = any(term in text for term in terms)
            if image_id not in chosen and matches:
                chosen.append(image_id)
                break
    for image_id in test_ids:
        if len(chosen) >= count:
            break
        if image_id not in chosen:
            chosen.append(image_id)
    return chosen[:count]


def evaluate(max_images: int = 10) -> list[dict]:
    from tensorflow.keras.models import load_model

    if not ATTENTION_FINAL_PATH.is_file():
        raise FileNotFoundError(f"Attention candidate not found: {ATTENTION_FINAL_PATH}")
    captions = load_json(PROCESSED_CAPTIONS_PATH)
    test_ids = load_json(SPLIT_PATH)["test"]
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = load_model(ATTENTION_FINAL_PATH, compile=False)
    chosen = diagnostic_images(test_ids, captions, max_images)
    rows = []
    refs_bleu, hyps_bleu = [], []
    for image_id in chosen:
        path = REGION_FEATURES_DIR / f"{image_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Held-out region feature missing: {path}")
        feature = np.load(path).astype(np.float32)
        generated = beam_search_decode(model, feature, tokenizer)
        refs = [caption_tokens_for_bleu(text) for text in captions.get(image_id, [])]
        refs = [ref for ref in refs if ref]
        refs_bleu.append(refs)
        hyps_bleu.append(generated.split())
        rows.append({"image": image_id, "generated": generated,
                     "references": [" ".join(ref) for ref in refs]})
        print(f"{image_id}\n  generated: {generated}\n  reference: {rows[-1]['references'][0] if refs else ''}")
    scores = _bleu_scores(refs_bleu, hyps_bleu)
    scores["num_test_images"] = len(rows)
    scores["note"] = "Diagnostic category-stratified subset; use full test evaluation before deployment."
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    dump_json(PREDICTIONS_DIR / "attention_test_predictions.json", rows)
    dump_json(PREDICTIONS_DIR / "attention_bleu_scores.json", scores)
    with (PREDICTIONS_DIR / "attention_test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "generated_caption", "actual_captions"])
        writer.writerows([[r["image"], r["generated"], " || ".join(r["references"])] for r in rows])
    print(f"Diagnostic BLEU-1={scores['bleu1']:.4f}, BLEU-4={scores['bleu4']:.4f}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=10)
    evaluate(max(1, parser.parse_args().max_images))

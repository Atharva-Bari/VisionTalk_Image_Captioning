"""Regression checks for caption corruption and decoder end-token handling."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import _pick_next_id_without_repeat, caption_quality_issues
from src.feature_extraction import (
    _ensure_metadata_compatible,
    _normalize_features,
    expected_extractor_metadata,
)


def test_accepts_a_short_natural_caption() -> None:
    assert caption_quality_issues("A dog runs through the grass") == []


def test_rejects_word_salad_from_sampling_tail() -> None:
    caption = "snow soldier snow bug snow waiter snow trophy snow medals snow dj snow rv"
    issues = caption_quality_issues(caption)
    assert "caption lacks basic sentence structure" in issues
    assert "caption repeats a word excessively" in issues


def test_rejects_repeated_caption() -> None:
    issues = caption_quality_issues("person person person person person")
    assert "caption repeats a word excessively" in issues
    assert "caption has too little vocabulary diversity" in issues


def test_end_token_is_not_suppressed_when_ranked_first() -> None:
    probs = np.asarray([0.01, 0.9, 0.09], dtype=np.float32)
    assert _pick_next_id_without_repeat(probs, [], end_id=1) == 1


def test_vgg_feature_normalization_matches_checkpoint_input_contract() -> None:
    features = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = _normalize_features(features)
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.allclose(normalized[1], [0.0, 0.0])


def test_stale_feature_cache_metadata_is_rejected() -> None:
    expected = expected_extractor_metadata(4096)
    stale = dict(expected)
    stale.update(include_top=False, pooling="avg", feature_vector_dim=512)
    try:
        _ensure_metadata_compatible(stale, expected, force=False)
    except ValueError as exc:
        assert "include_top" in str(exc)
    else:
        raise AssertionError("512-D pooled cache metadata should not be accepted for the 4096-D model")


if __name__ == "__main__":
    test_accepts_a_short_natural_caption()
    test_rejects_word_salad_from_sampling_tail()
    test_rejects_repeated_caption()
    test_end_token_is_not_suppressed_when_ranked_first()
    test_vgg_feature_normalization_matches_checkpoint_input_contract()
    test_stale_feature_cache_metadata_is_rejected()
    print("caption quality regression checks passed")

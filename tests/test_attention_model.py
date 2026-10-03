import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attention_model import build_attention_caption_model
from src.inference import _model_feature_batch, _predict_next_probs


def test_attention_captioner_uses_spatial_grid_and_normalizes_attention():
    model = build_attention_caption_model(
        vocab_size=24,
        max_length=8,
        region_count=49,
        region_dim=512,
        embedding_dim=8,
        lstm_units=8,
        attention_dim=8,
        dropout=0.0,
    )
    regions = np.random.default_rng(3).normal(size=(2, 49, 512)).astype(np.float32)
    tokens = np.zeros((2, 8), dtype=np.int32)
    tokens[:, 0] = 2
    predictions = model.predict([regions, tokens], verbose=0)
    attention_model = __import__("tensorflow").keras.Model(
        model.inputs, model.get_layer("spatial_attention_weights").output
    )
    weights = attention_model.predict([regions, tokens], verbose=0)

    assert model.input_shape[0] == (None, 49, 512)
    assert predictions.shape == (2, 8, 24)
    assert np.isfinite(predictions).all()
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
    assert not np.allclose(predictions[0], predictions[1])
    assert _model_feature_batch(regions[0]).shape == (1, 49, 512)
    assert np.allclose(_predict_next_probs(model, regions[:1], tokens[:1], 2), predictions[0, -1])


if __name__ == "__main__":
    test_attention_captioner_uses_spatial_grid_and_normalizes_attention()
    print("attention model checks passed")

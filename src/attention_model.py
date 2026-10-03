"""Spatial-attention VGG16 + LSTM captioner used by the improved pipeline."""

from __future__ import annotations

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Dense,
    Dot,
    Dropout,
    Embedding,
    Input,
    LSTM,
    LayerNormalization,
    Reshape,
    Softmax,
)
from tensorflow.keras.optimizers import Adam


def build_attention_caption_model(
    vocab_size: int,
    max_length: int,
    *,
    region_count: int = 49,
    region_dim: int = 512,
    embedding_dim: int = 256,
    lstm_units: int = 256,
    attention_dim: int = 256,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """Build a captioner whose word prediction attends to VGG16 image regions."""
    regions = Input((region_count, region_dim), name="image_regions")
    normalized = LayerNormalization(axis=-1, name="region_norm")(regions)
    keys = Dense(attention_dim, activation="tanh", name="region_keys")(normalized)

    tokens = Input((max_length,), dtype="int32", name="caption_sequence")
    embedded = Embedding(vocab_size, embedding_dim, mask_zero=False, name="caption_embedding")(tokens)
    # The model learns a visual context for every output word. `mask` keeps
    # post-padding out of the recurrent state, and the attention map is
    # computed independently at each decoder timestep.
    hidden = LSTM(lstm_units, return_sequences=True, name="caption_lstm")(
        embedded, mask=tokens != 0
    )
    query = Dense(attention_dim, name="caption_query")(hidden)
    keys_by_step = Reshape((1, region_count, attention_dim), name="keys_by_step")(keys)
    query_by_region = Reshape((max_length, 1, attention_dim), name="query_by_region")(query)
    energy = Add(name="attention_fusion")([keys_by_step, query_by_region])
    scores = Dense(1, activation="tanh", name="attention_scores")(energy)
    scores = Reshape((max_length, region_count), name="attention_logits")(scores)
    weights = Softmax(axis=-1, name="spatial_attention_weights")(scores)
    context = Dot(axes=(2, 1), name="attention_context")([weights, keys])

    fused = Concatenate(name="visual_text_fusion")([context, hidden])
    fused = Dense(256, activation="relu", name="fusion_dense")(fused)
    fused = Dropout(dropout, name="fusion_dropout")(fused)
    output = Dense(vocab_size, activation="softmax", name="next_word")(fused)
    model = Model([regions, tokens], output, name="visiontalk_vgg16_spatial_attention")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=["accuracy"],
    )
    return model

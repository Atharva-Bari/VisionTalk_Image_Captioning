"""CNN-feature + LSTM captioning model (Keras Functional API).

Importing this module does not start training.
"""

from __future__ import annotations

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
    LeakyReLU,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal

from src.config import (
    DENSE_UNITS,
    DROPOUT,
    EMBEDDING_DIM,
    FEATURE_VECTOR_DIM,
    LEARNING_RATE,
    LSTM_UNITS,
)


def build_caption_model(
    vocab_size: int,
    max_length: int,
    feature_dim: int = FEATURE_VECTOR_DIM,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout: float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """Two-input next-word model: image features + partial caption sequence."""
    if vocab_size < 4:
        raise ValueError(f"vocab_size must be at least 4, got {vocab_size}.")
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2, got {max_length}.")

    image_input = Input(shape=(feature_dim,), name="image_features")
    image_bn = BatchNormalization(name="image_bn")(image_input)
    image_dense = Dense(
        dense_units,
        kernel_initializer=HeNormal(),
        use_bias=True,
        name="image_dense",
    )(image_bn)
    image_act = LeakyReLU(negative_slope=0.1, name="image_leaky_relu")(image_dense)
    image_drop = Dropout(dropout, name="image_dropout")(image_act)

    caption_input = Input(shape=(max_length,), name="caption_sequence")
    embedded = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="caption_embedding",
    )(caption_input)
    lstm_out = LSTM(lstm_units, name="caption_lstm")(embedded)

    merged = Concatenate(name="merge")([image_drop, lstm_out])
    fused = Dense(
        dense_units,
        kernel_initializer=HeNormal(),
        name="fusion_dense",
    )(merged)
    fused = LeakyReLU(negative_slope=0.1, name="fusion_leaky_relu")(fused)
    fused = Dropout(dropout, name="fusion_dropout")(fused)
    output = Dense(vocab_size, activation="softmax", name="next_word")(fused)

    model = Model(inputs=[image_input, caption_input], outputs=output, name="visiontalk_cnn_lstm")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def dummy_forward_pass(
    model: Model | None = None,
    vocab_size: int = 32,
    max_length: int = 8,
    feature_dim: int = 16,
    batch_size: int = 2,
):
    """Run one dummy batch to verify output rank and vocabulary dimension."""
    import numpy as np

    if model is None:
        model = build_caption_model(
            vocab_size=vocab_size,
            max_length=max_length,
            feature_dim=feature_dim,
            embedding_dim=8,
            lstm_units=8,
            dense_units=8,
            dropout=0.0,
        )
    images = np.zeros((batch_size, int(model.inputs[0].shape[-1])), dtype=np.float32)
    sequences = np.zeros((batch_size, int(model.inputs[1].shape[1])), dtype=np.int32)
    sequences[:, 0] = 2
    predictions = model.predict([images, sequences], verbose=0)
    expected_vocab = model.output_shape[-1]
    if predictions.shape != (batch_size, expected_vocab):
        raise RuntimeError(
            f"Dummy forward pass produced {predictions.shape}, "
            f"expected {(batch_size, expected_vocab)}."
        )
    return predictions


def build_model() -> Model:
    """Compatibility wrapper used by tests; uses tiny dummy sizes."""
    return build_caption_model(vocab_size=16, max_length=6, feature_dim=16)


if __name__ == "__main__":
    demo = build_caption_model(vocab_size=64, max_length=10, feature_dim=1280)
    demo.summary()
    dummy_forward_pass(demo, batch_size=1)
    print("Dummy forward pass OK.")

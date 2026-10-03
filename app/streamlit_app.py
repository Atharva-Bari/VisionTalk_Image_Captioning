from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CNN_MODEL_NAME,
    CNN_POOLING,
    FEATURE_VECTOR_DIM,
    FINAL_MODEL_PATH,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    TOKENIZER_PATH,
)
from src.feature_extraction import NORM_VGG16_FEATURES, build_encoder_for_dim
from src.inference import INFERENCE_CODE_VERSION, generate_caption, resolve_model_path
from src.tokenizer_utils import load_tokenizer

st.set_page_config(
    page_title="VisionTalk — Image Captioning",
    page_icon="🎙️",
    layout="centered",
)

st.title("VisionTalk Image Captioning")
st.caption("Upload any image → get an automatic caption.")


@st.cache_resource(show_spinner="Loading model...")
def load_runtime():
    from src.inference import load_caption_model

    resolved = resolve_model_path(FINAL_MODEL_PATH)
    model, _ = load_caption_model(resolved)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    expected_dim = int(model.inputs[0].shape[-1])
    if int(model.output_shape[-1]) != int(tokenizer["vocab_size"]):
        raise ValueError(
            f"Model vocabulary dimension {model.output_shape[-1]} does not match tokenizer "
            f"vocabulary size {tokenizer['vocab_size']}. Use the tokenizer saved with this model."
        )
    word_to_index = tokenizer["word_to_index"]
    index_to_word = {int(index): word for index, word in tokenizer["index_to_word"].items()}
    if any(index_to_word.get(int(index)) != word for word, index in word_to_index.items()):
        raise ValueError("Tokenizer word/id mappings are inconsistent; refusing to decode model output.")
    if len(model.inputs) < 2 or int(model.inputs[1].shape[-1]) != int(tokenizer["max_caption_length"]):
        raise ValueError("Model caption-sequence length does not match the saved tokenizer metadata.")
    encoder = build_encoder_for_dim(expected_dim)
    actual_dim = int(encoder.output_shape[-1])
    if actual_dim != expected_dim:
        raise ValueError(f"Encoder outputs {actual_dim} features, but the model expects {expected_dim}.")
    return model, tokenizer, encoder, resolved, expected_dim


try:
    model, tokenizer, encoder, model_path, expected_dim = load_runtime()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Captioning runtime failed validation: {exc}")
    st.stop()

with st.expander("Model Diagnostics"):
    st.write({
        "model loaded": "YES",
        "tokenizer loaded": "YES",
        "encoder loaded": "YES",
        "model file": str(model_path),
        "tokenizer file": str(TOKENIZER_PATH),
        "model image input dimension": expected_dim,
        "encoder feature dimension": int(encoder.output_shape[-1]),
        "model text input length": int(model.inputs[1].shape[-1]),
        "vocabulary size": int(tokenizer["vocab_size"]),
        "configured feature dimension": FEATURE_VECTOR_DIM,
        "feature normalization": NORM_VGG16_FEATURES,
        "CNN encoder": f"{CNN_MODEL_NAME} ({CNN_POOLING}), image size={IMAGE_SIZE}",
        "inference function": "src.inference.generate_caption",
        "decoder": "beam search (quality-gated; greedy retry)",
        "inference code version": INFERENCE_CODE_VERSION,
    })

uploaded = st.file_uploader(
    "Upload an image",
    type=sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS),
    help="Supports JPEG and PNG images.",
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (OSError, ValueError) as exc:
        st.error(f"Could not open the uploaded image: {exc}")
        image = None
else:
    image = None

if image is not None:
    st.image(image, caption="Preview", use_container_width=True)
    if st.button("Generate Caption", type="primary"):
        temp_path = ROOT / "outputs" / "predictions" / "_upload.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(temp_path, format="JPEG")
        with st.spinner("Generating caption..."):
            try:
                caption = generate_caption(
                    temp_path,
                    model=model,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    decoder="beam",
                )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
                caption = None
        if caption:
            st.subheader("Generated caption")
            st.markdown(f"> {caption}")
else:
    st.info("Upload an image above, then click **Generate Caption**.")

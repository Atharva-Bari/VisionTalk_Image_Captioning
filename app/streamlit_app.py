"""VisionTalk – AI Image Caption Generator.

Loads a trained model and tokenizer only. Does not train, download data,
rebuild the vocabulary, or extract the Flickr8k feature cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FINAL_MODEL_PATH, IMAGE_EXTENSIONS, TOKENIZER_PATH
from src.feature_extraction import build_encoder
from src.inference import generate_caption, resolve_model_path
from src.tokenizer_utils import load_tokenizer


@st.cache_resource
def load_runtime():
    model_path = resolve_model_path()
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    encoder = build_encoder()
    return model, tokenizer, encoder, model_path


def main() -> None:
    st.set_page_config(page_title="VisionTalk – AI Image Caption Generator", layout="centered")
    st.title("VisionTalk – AI Image Caption Generator")
    st.caption(
        "CNN encoder (VGG16) + LSTM decoder. This is a research prototype, "
        "not a production assistive device."
    )

    if not TOKENIZER_PATH.is_file():
        st.error("Tokenizer not found. Run: python -m src.text_preprocessing")
        return
    try:
        resolve_model_path()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model before using this app: python -m src.train")
        return

    try:
        model, tokenizer, encoder, model_path = load_runtime()
    except (OSError, ValueError, FileNotFoundError) as exc:
        st.error(f"Could not load the captioning runtime: {exc}")
        return

    st.success(f"Loaded model: {model_path.name}")
    uploaded = st.file_uploader(
        "Upload an image",
        type=[ext.lstrip(".") for ext in sorted(IMAGE_EXTENSIONS)],
    )
    if uploaded is None:
        st.info("Choose a JPEG or PNG image, then click Generate Caption.")
        return

    from PIL import Image

    try:
        image = Image.open(uploaded).convert("RGB")
    except OSError:
        st.error("Could not read the uploaded file. Use a valid JPEG or PNG image.")
        return

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
                )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
        st.subheader("Generated caption")
        st.write(caption)


if __name__ == "__main__":
    main()

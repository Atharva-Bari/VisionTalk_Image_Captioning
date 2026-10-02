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
from src.feature_extraction import build_encoder, build_encoder_for_dim
from src.inference import generate_caption, resolve_model_path
from src.tokenizer_utils import load_tokenizer


@st.cache_resource(show_spinner="Loading model and encoder…")
def load_runtime():
    model_path = resolve_model_path()
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    expected_dim = int(model.inputs[0].shape[-1])
    encoder = build_encoder_for_dim(expected_dim)

    actual_dim = int(encoder.output_shape[-1])
    if actual_dim != expected_dim:
        st.cache_resource.clear()
        raise ValueError(
            f"Dimension mismatch detected and cache cleared. "
            f"Encoder={actual_dim}, Model={expected_dim}. "
            "Please refresh the page or restart the Streamlit server once."
        )
    return model, tokenizer, encoder, model_path, expected_dim


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
        model, tokenizer, encoder, model_path, expected_dim = load_runtime()
    except (OSError, ValueError, FileNotFoundError) as exc:
        st.error(f"Could not load the captioning runtime: {exc}")
        if "Dimension mismatch" in str(exc):
            st.info("🔄 Press R or click 'Rerun' in the top-right to reload with the new encoder.")
        return

    with st.sidebar:
        st.header("Runtime")
        st.info(f"Model: `{model_path.name}`")
        encoder_dim = int(encoder.output_shape[-1])
        st.metric("Feature dim (encoder)", encoder_dim)
        st.metric("Feature dim (model)", expected_dim)
        if st.button("🔁 Clear cached model/encoder"):
            st.cache_resource.clear()
            st.success("Cache cleared. Rerun the page (press R).")
            st.stop()

        st.divider()
        st.header("Decoding")
        use_beam = st.toggle("Use beam search (recommended)", value=True,
                             help="Beam search explores multiple caption candidates and usually avoids the repetition loops that plague greedy decoding.")
        beam_size = st.slider("Beam size", min_value=2, max_value=7, value=3, step=1,
                              disabled=not use_beam)
        if use_beam:
            st.caption(f"Beam size {beam_size} · higher = more candidates, slower")
        else:
            st.caption("Greedy decoding · fastest but can still degenerate with very short captions")

    if encoder_dim == expected_dim:
        st.success(f"Loaded model: {model_path.name}  ·  features={encoder_dim}D")
    else:
        st.warning(f"Encoder dim {encoder_dim} ≠ model dim {expected_dim}. Clearing cache…")
        st.cache_resource.clear()
        st.stop()
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
        decoder_label = f"beam search (k={beam_size})" if use_beam else "greedy"
        with st.spinner(f"Generating caption with {decoder_label}..."):
            try:
                caption = generate_caption(
                    temp_path,
                    model=model,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    use_beam=use_beam,
                    beam_size=beam_size,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
        st.subheader("Generated caption")
        st.markdown(f"> {caption}")
        st.caption(f"Decoder: {decoder_label}")


if __name__ == "__main__":
    main()

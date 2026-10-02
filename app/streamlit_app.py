from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FINAL_MODEL_PATH, IMAGE_EXTENSIONS, TOKENIZER_PATH
from src.feature_extraction import build_encoder, build_encoder_for_dim
from src.inference import (
    INFERENCE_CODE_VERSION,
    generate_caption,
    generate_multiple_captions,
    resolve_model_path,
)
from src.tokenizer_utils import load_tokenizer

MAX_UPLOAD_MB = 50
st.set_page_config(
    page_title="VisionTalk — Image Captioning",
    page_icon="🎙️",
    layout="wide",
)

st.title("VisionTalk Image Captioning")
st.caption("VGG16 encoder + LSTM decoder · Flickr8k · Infer only (no training here).")


@st.cache_resource(show_spinner="Loading model & tokenizer...")
def load_runtime(model_path_input: Path | None = None):
    from src.inference import load_caption_model

    resolved = resolve_model_path(model_path_input)
    model, real_path = load_caption_model(resolved)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    expected_dim = int(model.inputs[0].shape[-1])
    encoder = build_encoder_for_dim(expected_dim)
    return model, tokenizer, encoder, expected_dim, real_path


try:
    model, tokenizer, encoder, expected_dim, model_path = load_runtime(FINAL_MODEL_PATH)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

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
    decoder_label_map = {
        "nucleus (recommended)": "nucleus",
        "beam search": "beam",
        "top-k sampling": "topk",
        "greedy (fastest)": "greedy",
    }
    decoder_display = st.selectbox(
        "Algorithm",
        list(decoder_label_map.keys()),
        index=0,
        help="Nucleus sampling is strongly recommended for this small model — greedy/beam collapse to single filler tokens like 'snow' or 'behind'.",
    )
    decoder = decoder_label_map[decoder_display]

    beam_size = st.slider("Beam size", min_value=2, max_value=7, value=3, step=1,
                          disabled=decoder != "beam")
    temperature = st.slider(
        "Temperature", min_value=0.3, max_value=1.6, value=0.9, step=0.05,
        disabled=decoder not in {"nucleus", "topk"},
        help="Higher = more creative / diverse; lower = more conservative.",
    )
    if decoder == "nucleus":
        p = st.slider("Nucleus (top-p)", min_value=0.5, max_value=0.99, value=0.9, step=0.01)
        k = 10
    elif decoder == "topk":
        k = st.slider("Top-k", min_value=2, max_value=50, value=10, step=1)
        p = 0.9
    else:
        p, k = 0.9, 10

    num_captions = st.slider("Number of captions", min_value=1, max_value=5, value=3, step=1,
                             help="Generate multiple candidate captions and pick the best one. Nucleus/top-k give true diversity.")

    if decoder == "nucleus":
        st.caption(f"Nucleus p={p:.2f}, T={temperature:.2f} · best balance of quality + creativity")
    elif decoder == "topk":
        st.caption(f"Top-k k={k}, T={temperature:.2f} · maximum diversity")
    elif decoder == "beam":
        st.caption(f"Beam size {beam_size} · deterministic, safest only if model is well-trained")
    else:
        st.caption("Greedy · fastest, very likely to output a single repeated/collapsed word")

    st.divider()
    st.header("About")
    st.caption(f"Inference code: `{INFERENCE_CODE_VERSION}`")
    with st.expander("Still seeing 'snow snow' or 'behind'?"):
        st.markdown(
            "1. Confirm the sidebar shows `v3.0-nucleus-ngram` above.  If not, click the app "
            "menu ⋯ → **Reboot app** to clear Python's sys.modules cache.\n\n"
            "2. Make sure **Algorithm = nucleus (recommended)** (first option in the dropdown).\n\n"
            "3. Increase Temperature to ~1.1 and Number of captions to 3 — one of the candidates "
            "will almost always be a proper full sentence.\n\n"
            "4. For very challenging photos (mountains, city streets with unusual objects), this "
            "small Flickr8k-trained model will never be perfect.  Nucleus + multiple candidates "
            "is the best practical recovery without retraining."
        )

col1, col2 = st.columns([1.3, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload an image",
        type=sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS),
        help=f"Max {MAX_UPLOAD_MB} MB per image.",
    )
    camera = st.camera_input("…or take a photo", disabled=False,
                             help="Optional live camera capture (same pipeline).")

image: Image.Image | None = None
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (OSError, ValueError) as exc:
        st.error(f"Could not open the uploaded image: {exc}")
        image = None
elif camera is not None:
    try:
        image = Image.open(camera).convert("RGB")
    except (OSError, ValueError) as exc:
        st.error(f"Could not open the camera capture: {exc}")
        image = None

with col2:
    if image is None:
        st.info("Upload an image on the left, or use the camera, then click **Generate Caption**.")
    else:
        st.image(image, caption="Preview", use_container_width=True)

st.divider()

if image is not None:
    action_label = (
        f"Generate {num_captions} Caption" + ("s" if num_captions != 1 else "") +
        f" · {decoder_display.split(' (')[0]}"
    )
    if st.button(action_label, type="primary", use_container_width=True):
        temp_path = ROOT / "outputs" / "predictions" / "_upload.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(temp_path, format="JPEG")
        decoder_name = decoder_display.split(" (")[0]
        with st.spinner(f"Generating caption(s) with {decoder_name}..."):
            try:
                if num_captions == 1:
                    captions = [
                        generate_caption(
                            temp_path,
                            model=model,
                            tokenizer=tokenizer,
                            encoder=encoder,
                            decoder=decoder,
                            beam_size=beam_size,
                            p=p,
                            k=k,
                            temperature=temperature,
                        )
                    ]
                else:
                    captions = generate_multiple_captions(
                        temp_path,
                        model=model,
                        tokenizer=tokenizer,
                        encoder=encoder,
                        n=num_captions,
                        decoder=decoder,
                        beam_size=beam_size,
                        p=p,
                        k=k,
                        temperature=temperature,
                    )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
                captions = []
        if captions:
            st.subheader("Generated caption" + ("s" if len(captions) > 1 else ""))
            for i, caption in enumerate(captions, 1):
                if len(captions) == 1:
                    st.markdown(f"> {caption}")
                else:
                    with st.expander(f"Caption {i} 📌", expanded=(i == 1)):
                        st.markdown(f"**{caption}**")
            st.caption(
                f"Decoder: {decoder_name}" +
                (f" · p={p:.2f}, T={temperature:.2f}" if decoder in {"nucleus", "topk"} else
                 f" · beam size {beam_size}" if decoder == "beam" else "") +
                f" · code `{INFERENCE_CODE_VERSION}`"
            )
            if any(len(c.split()) <= 3 for c in captions):
                st.warning(
                    "One or more captions are very short (≤3 words).  Switch Algorithm to "
                    "**Nucleus (recommended)**, bump Temperature above 1.0, and try generating "
                    "3 captions — one will almost certainly be a proper sentence."
                )

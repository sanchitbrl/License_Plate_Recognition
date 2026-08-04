
import io
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import pipeline as pl

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "plate_detector.pt")
CHAR_SEG_MODEL_PATH = os.path.join(MODELS_DIR, "char_segmenter.pt")
CHAR_CNN_PATH = os.path.join(MODELS_DIR, "char_cnn_checkpoint.pth")


PLATE_CONF = 0.5
CHAR_CONF = 0.25
PAD_RATIO = 0.08

DO_ENHANCE = True
TARGET_MIN_HEIGHT = 180
DO_CLAHE = True
DO_SHARPEN = True


@st.cache_resource(show_spinner="Loading plate detector...")
def load_plate_model():
    return pl.load_plate_model(PLATE_MODEL_PATH)


@st.cache_resource(show_spinner="Loading character segmenter...")
def load_char_seg_model():
    return pl.load_char_seg_model(CHAR_SEG_MODEL_PATH)


@st.cache_resource(show_spinner="Loading character classifier...")
def load_char_cnn():
    return pl.load_char_cnn(CHAR_CNN_PATH)


def main():
    st.set_page_config(page_title="Nepali License Plate Recognition", layout="wide")
    st.title(" Nepali License Plate Recognition")
    st.caption("Upload an image → plate detection → character segmentation → character recognition")

    missing = [p for p in [PLATE_MODEL_PATH, CHAR_SEG_MODEL_PATH, CHAR_CNN_PATH] if not os.path.exists(p)]
    if missing:
        st.error(
            "Missing model file(s). Place these in the `models/` folder next to app.py:\n\n"
            + "\n".join(f"- `{os.path.basename(p)}`" for p in missing)
        )
        st.stop()

    uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload an image to run the pipeline.")
        return

    image_pil = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(image_pil, use_container_width=True)

    plate_model = load_plate_model()
    char_seg_model = load_char_seg_model()
    cnn_model, class_names, cnn_transform = load_char_cnn()

    with st.spinner("Detecting plate..."):
        plate_crop, box = pl.detect_plate(plate_model, image_bgr, PLATE_CONF, pad_ratio=PAD_RATIO)

    if plate_crop is None:
        st.warning("No license plate detected.")
        return

    if DO_ENHANCE:
        enhanced_crop = pl.enhance_plate_crop(
            plate_crop,
            target_min_height=TARGET_MIN_HEIGHT,
            apply_clahe=DO_CLAHE,
            apply_sharpen=DO_SHARPEN,
        )
    else:
        enhanced_crop = plate_crop

    with col2:
        st.subheader("Detected plate")
        sub1, sub2 = st.columns(2)
        sub1.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption="Raw crop", use_container_width=True)
        sub2.image(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB), caption="Enhanced", use_container_width=True)

    with st.spinner("Segmenting characters..."):
        char_crops = pl.segment_characters(char_seg_model, enhanced_crop, CHAR_CONF)

    if not char_crops:
        st.warning("No characters detected on the plate.")
        return

    st.subheader(f"Segmented characters ({len(char_crops)}) — reading order left→right, top row first")
    char_cols = st.columns(len(char_crops))

    with st.spinner("Reading characters..."):
        predicted_chars = pl.classify_characters(cnn_model, class_names, cnn_transform, char_crops)

    for i, (c, crop, pred) in enumerate(zip(char_cols, char_crops, predicted_chars)):
        c.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"#{i+1}: {pred}", use_container_width=True)

    formatted, raw = pl.format_plate(predicted_chars)

    st.divider()
    st.subheader("Result")
    st.metric("Detected plate", formatted)
    if formatted != raw:
        st.caption(f"Raw character sequence: {raw}")


if __name__ == "__main__":
    main()
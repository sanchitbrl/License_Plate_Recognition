import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import time

# page setup
st.set_page_config(
    page_title="Nepali LPR System",
    page_icon="",
    layout="centered",
)

# styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .main {
        background-color: #0d0f14;
        color: #e8eaf0;
    }

    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
    }

    .plate-box {
        background: linear-gradient(135deg, #1a1d27, #12151f);
        border: 2px solid #3af0a2;
        border-radius: 12px;
        padding: 24px 32px;
        text-align: center;
        margin: 16px 0;
    }

    .plate-text {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: #3af0a2;
        letter-spacing: 0.15em;
    }

    .raw-ocr {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #7a8099;
        margin-top: 8px;
    }

    .step-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #7a8099;
        margin-bottom: 4px;
    }

    .info-chip {
        display: inline-block;
        background: #1a1d27;
        border: 1px solid #2a2e3f;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: #9aa0b8;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("# Nepali LPR")
st.markdown("**License Plate Recognition** — YOLOv8 + EasyOCR (Nepali)")
st.markdown("---")

# load models once
@st.cache_resource
def load_models():
    model = YOLO("best(LP dectection).pt")
    model.model.names[0] = "license_plate"
    reader = easyocr.Reader(["ne"], gpu=False)
    return model, reader

with st.spinner("Loading models..."):
    model, reader = load_models()

st.success("Models loaded")

# file upload
st.markdown("### Upload Vehicle Image")
uploaded_file = st.file_uploader(
    "Supports JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# nepali plate characters
ALLOWLIST    = "०१२३४५६७८९बामेकोसजनागलुधराभेकसेमपप्रखफझबघञया"
VALID_CHARS  = "बामेकोसजनागलुधराभेकसेमपप्रखफझबघञया"
VALID_DIGITS = "०१२३४५६७८९"


def preprocess_plate(crop):
    # upscale, grayscale, blur, threshold, sharpen
    plate = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharp  = cv2.filter2D(thresh, -1, kernel)
    return crop, sharp


def format_plate(full_text):
    # split into letters and digits then rebuild plate format
    letters, numbers = [], []
    for ch in full_text:
        if ch in VALID_CHARS:    letters.append(ch)
        elif ch in VALID_DIGITS: numbers.append(ch)

    first_char = letters[0]  if len(letters) > 0 else ""
    middle_num = "".join(numbers[:2])  if len(numbers) >= 2 else "".join(numbers)
    last_char  = letters[-1] if len(letters) > 1 else ""
    last_4     = "".join(numbers[-4:]) if len(numbers) >= 4 else "".join(numbers[2:])
    return first_char + middle_num + last_char + " " + last_4


if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # show input
    st.markdown("### Step 1 — Input Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # run yolo
    st.markdown("### Step 2 — Plate Detection")
    with st.spinner("Running YOLOv8..."):
        results = model.predict(source=img, conf=0.5, verbose=False)
        time.sleep(0.3)

    plates_found = False

    for result in results:
        for i, box in enumerate(result.boxes):
            plates_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = float(box.conf[0])

            # draw box on image
            annotated = img.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (58, 240, 162), 3)
            cv2.putText(
                annotated,
                f"plate  {conf_score:.0%}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (58, 240, 162),
                2,
            )
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detection confidence: **{conf_score:.1%}**")

            # preprocess crop
            st.markdown("### Step 3 — Preprocessing")
            crop = img[y1:y2, x1:x2]
            original_crop, sharp = preprocess_plate(crop)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="step-label">Cropped Plate</p>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown('<p class="step-label">Processed</p>', unsafe_allow_html=True)
                st.image(sharp, use_container_width=True)

            # run ocr
            st.markdown("### Step 4 — OCR")
            cv2.imwrite("_plate_temp.jpg", sharp)
            with st.spinner("Reading text..."):
                ocr_result = reader.readtext(
                    "_plate_temp.jpg",
                    detail=0,
                    paragraph=False,
                    allowlist=ALLOWLIST,
                )
            full_text = "".join(ocr_result)
            formatted = format_plate(full_text)

            # final output
            st.markdown("### Result")
            st.markdown(
                f"""
                <div class="plate-box">
                    <div class="plate-text">{formatted if formatted.strip() else "could not read"}</div>
                    <div class="raw-ocr">raw ocr: {full_text if full_text else "nothing detected"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not plates_found:
        st.warning("No license plate detected. Try a clearer image.")

# footer
st.markdown("---")
st.markdown(
    '<span class="info-chip">YOLOv8s</span>'
    '<span class="info-chip">EasyOCR Nepali</span>'
    '<span class="info-chip">OpenCV</span>',
    unsafe_allow_html=True,
)

# Nepali License Plate Recognition — Technical Documentation

## 1. Overview

This document provides a complete technical description of the Nepali License Plate Recognition (LPR) system: its architecture, models, algorithms, code structure, and the design decisions behind it — including why the initial EasyOCR-based approach was replaced with a custom-trained CNN classifier.

The system takes a photograph of a vehicle as input and outputs the recognized license plate string in standard Nepali plate format (e.g. `बा२ख 1234`). It is composed of two independently trained detection models and (in the final version) one custom-trained classification model, orchestrated through a shared pipeline.

---

## 2. System Architecture

The project evolved through two generations, both present in this repository:

| | Generation 1 (`app.py`) | Generation 2 (`app2.py` + `pipeline.py`) |
|---|---|---|
| Plate detection | YOLOv8 | YOLOv8 |
| Character extraction | EasyOCR (full-plate OCR) | YOLOv8 (per-character detection) |
| Character recognition | EasyOCR built-in recognizer | Custom-trained CNN (`CharCNN`) |
| Preprocessing | Grayscale + threshold + sharpen | Color-preserving CLAHE + unsharp mask |
| Models required | 1 (`.pt`) | 3 (`.pt`, `.pt`, `.pth`) |

Both generations share the same first stage (plate detection via YOLOv8) but diverge completely at the character-recognition stage. Generation 2 is the current recommended pipeline; Generation 1 is retained in the repo for comparison and historical reference.

### 2.1 High-level pipeline (Generation 2)

```
Input Image
    │
    ▼
[1] Plate Detection (YOLOv8)  ──► bounding box + crop
    │
    ▼
[2] Crop Enhancement (padding, upscale, CLAHE, unsharp mask)
    │
    ▼
[3] Character Segmentation (YOLOv8, second model)  ──► per-character bounding boxes
    │
    ▼
[4] Reading-order resolution (row-aware clustering + left-to-right sort)
    │
    ▼
[5] Character Classification (custom CNN, per crop)  ──► predicted character sequence
    │
    ▼
[6] Plate Formatting  ──► final plate string
```

---

## 3. Generation 1: YOLOv8 + EasyOCR

### 3.1 Rationale for the initial approach

The first working version of this system was built around **EasyOCR**, a general-purpose, pretrained OCR library that supports Nepali (`ne`) out of the box. It was chosen initially because it required no additional model training beyond the plate detector — recognition was delegated entirely to EasyOCR's pretrained recognition network.

### 3.2 Implementation

**Plate detection** uses a YOLOv8 model trained to detect a single class (license plates):

```python
@st.cache_resource
def load_models():
    model = YOLO("best(LP dectection).pt")
    model.model.names[0] = "license_plate"
    reader = easyocr.Reader(["ne"], gpu=False)
    return model, reader
```

**Preprocessing** applies classical image-processing operations to the cropped plate before OCR — upscaling, grayscale conversion, blur, binary inverse thresholding, and a sharpening convolution:

```python
def preprocess_plate(crop):
    plate = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharp  = cv2.filter2D(thresh, -1, kernel)
    return crop, sharp
```

**Recognition** runs EasyOCR on the processed crop with a character allowlist restricted to valid Devanagari digits and the letters that appear on Nepali plates:

```python
ALLOWLIST = "०१२३४५६७८९बामेकोसजनागलुधराभेकसेमपप्रखफझबघञया"

ocr_result = reader.readtext(
    "_plate_temp.jpg",
    detail=0,
    paragraph=False,
    allowlist=ALLOWLIST,
)
full_text = "".join(ocr_result)
```

**Plate reconstruction** separates the raw OCR output into letters and digits, then reassembles it according to the assumed `[char][2-digit zone][char] [4-digit number]` layout:

```python
def format_plate(full_text):
    letters, numbers = [], []
    for ch in full_text:
        if ch in VALID_CHARS:    letters.append(ch)
        elif ch in VALID_DIGITS: numbers.append(ch)

    first_char = letters[0]  if len(letters) > 0 else ""
    middle_num = "".join(numbers[:2])  if len(numbers) >= 2 else "".join(numbers)
    last_char  = letters[-1] if len(letters) > 1 else ""
    last_4     = "".join(numbers[-4:]) if len(numbers) >= 4 else "".join(numbers[2:])
    return first_char + middle_num + last_char + " " + last_4
```

### 3.3 Observed limitations

Once tested against a broader and more realistic set of plate images, EasyOCR's shortcomings became clear:

- **EasyOCR is a generalized, multi-language OCR engine.** Its Nepali recognition weights are trained to read Devanagari script broadly — signage, documents, printed text in general — not specifically the constrained, stylized character set and fixed layout used on Nepali vehicle plates. This generalization comes at the cost of precision on this narrow, high-stakes task.
- **No domain-specific fine-tuning.** Because EasyOCR's model is not trained on license-plate imagery specifically, it does not learn plate-specific character shapes, spacing, or the embossed/stamped font styles commonly used on plates.
- **Sensitivity to real-world conditions.** Accuracy degraded significantly under glare, uneven lighting, motion blur, oblique viewing angles, and physical plate wear — all common in real vehicle photographs, as opposed to the clean, front-facing document scans OCR engines are typically evaluated on.
- **Character-level errors cascade into formatting errors.** Since `format_plate()` depends on getting the correct count and order of letters/digits, even a single misread or dropped character shifts the entire reconstruction, corrupting the whole plate string rather than just one character.

This combination of a generalized model applied to a specialized problem was the direct motivation for Generation 2: rather than relying on a pretrained, general-purpose recognizer, the team trained a character classifier specifically on Nepali plate character crops.

---

## 4. Generation 2: YOLOv8 + Character Segmentation + Custom CNN

### 4.1 Design rationale

Generation 2 replaces the single "detect plate → OCR the whole thing" flow with a three-model pipeline where **every stage is trained specifically for this task**:

1. A YOLOv8 model trained only to find plates.
2. A second, separately trained YOLOv8 model whose only job is to find *individual characters* within a plate crop (i.e., detection, not recognition).
3. A custom CNN trained only to classify a single cropped character image into one of the known Nepali plate character classes (digits 0–9 and the specific letters used on Nepali plates).

By decomposing "read the plate" into "find each character" + "classify each character," each model has a narrow, well-defined job, which is both easier to train well and easier to debug when something goes wrong — a misread can be traced to a specific crop and a specific model prediction.

### 4.2 Plate detection and enhancement

Plate detection uses the same YOLOv8 approach as Generation 1, but with padding added around the detected box, since tight YOLO boxes were found to clip the first or last character on a plate:

```python
def pad_box(box, image_shape, pad_ratio=0.08):
    x1, y1, x2, y2 = box
    h, w = image_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, w)
    y2 = min(y2 + pad_y, h)
    return x1, y1, x2, y2


def detect_plate(plate_model, image_bgr, conf, pad_ratio=0.08):
    results = plate_model.predict(source=image_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])
    x1, y1, x2, y2 = pad_box((x1, y1, x2, y2), image_bgr.shape, pad_ratio=pad_ratio)
    crop = image_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)
```

Crop enhancement deliberately **stays in color** rather than binarizing, since the CNN classifier expects color input normalized the same way as its training data — thresholding to black-and-white here would push inputs off-distribution and hurt classification accuracy:

```python
def enhance_plate_crop(crop_bgr, target_min_height=180, max_scale=6.0,
                        apply_clahe=True, apply_sharpen=True):
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr

    scale = max(1.0, target_min_height / h)
    scale = min(scale, max_scale)
    if scale > 1.0:
        crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if apply_clahe:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        crop_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if apply_sharpen:
        blurred = cv2.GaussianBlur(crop_bgr, (0, 0), sigmaX=1.5)
        crop_bgr = cv2.addWeighted(crop_bgr, 1.5, blurred, -0.5, 0)

    return crop_bgr
```

- **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) is applied only to the L (lightness) channel in LAB color space, improving local contrast without distorting color information or over-amplifying noise in flat regions (the "contrast-limited" part caps how much any local region can be stretched).
- **Unsharp masking** (`addWeighted` with a blurred copy) increases edge definition, making character boundaries more distinct before segmentation.

### 4.3 Character segmentation

A second YOLOv8 model, trained specifically to detect individual characters (not plates), is run on the enhanced plate crop:

```python
def segment_characters(char_seg_model, plate_bgr, conf):
    results = char_seg_model.predict(source=plate_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    order = order_character_boxes(xyxy)

    crops = []
    for i in order:
        x1, y1, x2, y2 = map(int, xyxy[i])
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = plate_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    return crops
```

This model was trained separately (see `character_detection_model.py`), using a Roboflow-annotated dataset and reported the following validation metrics:

```
mAP50:     0.990
Precision: 0.984
Recall:    0.975
```

#### 4.3.1 Reading-order resolution

A naive left-to-right sort by x-coordinate fails whenever a plate has more than one line of text (common on Nepali plates, where a province designation may sit above the plate number) or is slightly rotated, since bounding-box x-ranges can then overlap across rows. This is solved with row-aware clustering:

```python
def order_character_boxes(xyxy):
    n = len(xyxy)
    if n == 0:
        return []
    if n == 1:
        return [0]

    y_centers = (xyxy[:, 1] + xyxy[:, 3]) / 2
    heights = xyxy[:, 3] - xyxy[:, 1]
    row_gap_thresh = max(np.median(heights) * 0.6, 1.0)

    order_by_y = np.argsort(y_centers)

    rows = [[order_by_y[0]]]
    for idx in order_by_y[1:]:
        prev_idx = rows[-1][-1]
        if abs(y_centers[idx] - y_centers[prev_idx]) > row_gap_thresh:
            rows.append([idx])
        else:
            rows[-1].append(idx)

    ordered_indices = []
    for row in rows:
        ordered_indices.extend(sorted(row, key=lambda i: xyxy[i][0]))
    return ordered_indices
```

The algorithm:
1. Computes each box's vertical center and sorts boxes by that value.
2. Walks through the y-sorted boxes, starting a new "row" whenever the gap in vertical center exceeds a threshold derived from the median character height (`0.6 × median height`) — this adapts automatically to different plate/character sizes rather than using a fixed pixel threshold.
3. Within each row, sorts boxes left-to-right by x-coordinate.
4. Concatenates rows top-to-bottom to produce the final reading order.

### 4.4 Custom CNN character classifier

This is the core addition of Generation 2, and directly addresses the accuracy problems observed with EasyOCR: instead of using a generalized OCR model, a convolutional neural network was trained from scratch, exclusively on cropped images of individual Nepali plate characters.

#### 4.4.1 Why a custom CNN

- **Narrow, well-defined task.** Unlike full-scale OCR (which must handle text detection, segmentation, and recognition across arbitrary layouts and scripts), this CNN only ever needs to answer one question: "which single character is in this crop?" A fixed set of known classes (Nepali digits 0–9 plus the specific letters used on plates) makes this a standard closed-set image classification problem — a much easier task than general-purpose scene text recognition.
- **Domain-specific training data.** The classifier is trained directly on crops produced by the character-segmentation model, from real plate images — meaning it learns the actual fonts, embossing/stamping styles, aspect ratios, and common degradations (dirt, glare, wear) seen on real Nepali plates, rather than generic printed-Devanagari examples.
- **Consistency with the rest of the pipeline.** Because the classifier's input distribution (crop size, color normalization, aspect ratio) is defined by the same pipeline that will feed it in production, there's no mismatch between training and inference conditions — a common source of accuracy loss when bolting a general-purpose external tool (like EasyOCR) onto a custom pipeline.

In practice, this specialization is what allowed the system to reach materially better accuracy than the EasyOCR-based approach, and to remain robust across the range of real-world conditions (lighting, angle, plate wear) that a generalized multi-language OCR model was not tuned to handle.

#### 4.4.2 Architecture

`CharCNN` is a compact convolutional network, defined in `pipeline.py`:

```python
class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Architecture breakdown:**

| Block | Layers | Output size (from 64×64 input) |
|---|---|---|
| Conv block 1 | 2× (Conv2d 3×3 → BatchNorm → ReLU), then MaxPool2d(2) | 32×32×32 |
| Conv block 2 | 2× (Conv2d 3×3 → BatchNorm → ReLU), then MaxPool2d(2) | 64×16×16 |
| Conv block 3 | 1× (Conv2d 3×3 → BatchNorm → ReLU), then MaxPool2d(2) | 128×8×8 |
| Classifier head | Flatten → Linear(8192→256) → ReLU → Dropout(0.4) → Linear(256→num_classes) | `num_classes` logits |

Design notes:
- **Double conv layers per early block** (blocks 1 and 2) increase representational capacity before downsampling, letting the network learn richer local features (character strokes, curves) at higher resolution before information is discarded by pooling.
- **Batch normalization** after every convolution stabilizes and accelerates training by normalizing activations, and reduces sensitivity to weight initialization — useful given the relatively small, domain-specific dataset this model is trained on.
- **Progressive channel widening** (3 → 32 → 64 → 128) is a standard CNN pattern: as spatial resolution shrinks, channel depth increases to preserve representational capacity.
- **Dropout (0.4)** in the classifier head is a regularization measure against overfitting, which is a real risk given a relatively narrow, specialized dataset of character crops.
- **Three pooling stages** (64→32→16→8) reduce a 64×64 input down to an 8×8 feature map before flattening, balancing spatial detail retention against parameter count in the fully connected layers.

#### 4.4.3 Preprocessing and inference

Character crops are converted from OpenCV's BGR format to RGB, wrapped as PIL images, and passed through a standard `torchvision` transform pipeline before inference:

```python
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
```

with:

```python
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
```

These are the standard ImageNet normalization statistics — used here as a sane default for RGB inputs even though the model is trained from scratch rather than fine-tuned from an ImageNet checkpoint.

Model loading restores weights and metadata from a checkpoint dictionary:

```python
def load_char_cnn(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", IMG_SIZE_DEFAULT)
    num_classes = checkpoint["num_classes"]

    model = CharCNN(num_classes).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return model, class_names, transform
```

Storing `class_names`, `img_size`, and `num_classes` inside the checkpoint (rather than hardcoding them in the inference code) keeps the model self-describing — the same loading code works regardless of how many character classes the model was trained on, or what input resolution was used.

Inference on a batch of segmented character crops:

```python
def classify_characters(cnn_model, class_names, transform, char_crops):
    predictions = []
    for crop_bgr in char_crops:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = cnn_model(tensor)
            pred_idx = int(logits.argmax(dim=1).item())
        predictions.append(class_names[pred_idx])
    return predictions
```

`torch.no_grad()` disables gradient tracking during inference, reducing memory use and speeding up the forward pass since no backpropagation is needed. `model.eval()` (set at load time) ensures BatchNorm and Dropout behave in inference mode (using running statistics / disabling dropout) rather than training mode.

Device selection is automatic:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

allowing the same code to run on either GPU or CPU without modification.

### 4.5 Plate formatting

The final stage reconstructs the plate string from the ordered sequence of predicted characters:

```python
def format_plate(chars):
    raw = "".join(chars)

    if len(chars) >= 4 and all(c in DEVANAGARI_DIGITS for c in chars[-4:]):
        prefix = "".join(chars[:-4])
        number = "".join(chars[-4:])
        return f"{prefix} {number}", raw

    return raw, raw
```

Unlike Generation 1's format function — which assumes a fixed-length zone code and can misalign when a plate's actual zone code is shorter than assumed — this version only asserts one fixed fact about Nepali plates: **the trailing 4 characters are always digits (the plate number)**. Everything before that is treated as an opaque prefix (province letter(s), zone digit(s), vehicle class letter) without trying to further subdivide it, since prefix structure varies by province and over-specifying it previously caused digits to be double-counted.

---

## 5. Application Layer (Streamlit)

### 5.1 `app2.py` structure

The Streamlit app is a thin UI layer over `pipeline.py`. Models are loaded once and cached across reruns using `st.cache_resource`:

```python
@st.cache_resource(show_spinner="Loading plate detector...")
def load_plate_model():
    return pl.load_plate_model(PLATE_MODEL_PATH)

@st.cache_resource(show_spinner="Loading character segmenter...")
def load_char_seg_model():
    return pl.load_char_seg_model(CHAR_SEG_MODEL_PATH)

@st.cache_resource(show_spinner="Loading character classifier...")
def load_char_cnn():
    return pl.load_char_cnn(CHAR_CNN_PATH)
```

This avoids reloading the YOLOv8 detectors and the CNN checkpoint on every user interaction (Streamlit reruns the whole script on each widget interaction by default).

The app performs an explicit startup check for required model files and fails early with a clear message rather than crashing deep inside the pipeline:

```python
missing = [p for p in [PLATE_MODEL_PATH, CHAR_SEG_MODEL_PATH, CHAR_CNN_PATH] if not os.path.exists(p)]
if missing:
    st.error(
        "Missing model file(s). Place these in the `models/` folder next to app.py:\n\n"
        + "\n".join(f"- `{os.path.basename(p)}`" for p in missing)
    )
    st.stop()
```

Key configuration constants at the top of the file control pipeline behavior without touching `pipeline.py`:

```python
PLATE_CONF = 0.5
CHAR_CONF = 0.25
PAD_RATIO = 0.08

DO_ENHANCE = True
TARGET_MIN_HEIGHT = 180
DO_CLAHE = True
DO_SHARPEN = True
```

### 5.2 `app.py` structure

The Generation 1 app follows a simpler, single-file structure (no separate pipeline module), with all detection, preprocessing, OCR, and formatting logic inline. It uses custom CSS injected via `st.markdown(..., unsafe_allow_html=True)` for a themed dark UI, and renders a bounding-box-annotated image, cropped/processed plate views, and the final result in a styled card.

---

## 6. Model Training (Reference Notebooks)

### 6.1 Plate detector (`LPR.py`)

Originally trained in Google Colab. Key steps:
- Dataset annotated and exported via Roboflow.
- Base model: `yolov8s.pt` (YOLOv8 small).
- Training invocation:

```python
model = YOLO("yolov8s.pt")
model.train(data='/content/LPR-1/data.yaml', epochs=20, imgsz=640)
```

- The resulting best checkpoint (`best(LP dectection).pt`) has its class name renamed post-load for clarity:

```python
model = YOLO('best(LP dectection).pt')
model.model.names[0] = 'License_plate'
```

This same notebook also contains the original prototype of the EasyOCR-based recognition and formatting logic that was later refactored into `app.py`.

### 6.2 Character detector (`character_detection_model.py`)

Also trained in Colab, using a separately annotated Roboflow dataset of individual plate characters:

```python
model = YOLO("yolov8s.pt")
model.train(data="/content/charcter-recognition-1/data.yaml", epochs=10, imgsz=640)
```

Validation metrics reported for this model:

```
mAP50:     0.990
Precision: 0.984
Recall:    0.975
```

### 6.3 Character CNN

The `CharCNN` training script itself is not included in this repository, but the checkpoint format it must produce is defined by `pipeline.py::load_char_cnn`, which expects a dictionary containing:
- `model_state_dict` — the trained weights, loadable into the `CharCNN` architecture defined above.
- `class_names` — an ordered list mapping output indices to character labels.
- `num_classes` — the number of output classes (must match `len(class_names)` and the classifier head's output dimension).
- `img_size` (optional, defaults to 64) — the square input resolution the model was trained on.

Training would have followed standard supervised image classification practice: character crops labeled by ground-truth class, split into train/validation sets, optimized with a cross-entropy loss against the `CharCNN` architecture's logits.

---

## 7. Dependencies

From `requirements.txt`:

```
streamlit
ultralytics
easyocr
opencv-python-headless
numpy
Pillow
```

Note: `torch` and `torchvision` are required by `pipeline.py` (Generation 2) but are not listed explicitly in `requirements.txt` — they are typically pulled in as a dependency of `ultralytics`, but should be pinned explicitly if reproducibility across environments matters.

---

## 8. File Reference

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — Generation 1 pipeline (YOLOv8 + EasyOCR) |
| `app2.py` | Streamlit app — Generation 2 pipeline (YOLOv8 + character segmentation + CNN) |
| `pipeline.py` | Core Generation 2 logic: model loaders, detection, enhancement, segmentation, ordering, classification, formatting |
| `LPR.py` | Original Colab notebook — plate detector training + Generation 1 OCR prototype |
| `character_detection_model.py` | Original Colab notebook — character detector training |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes model weights (`*.pt`, `*.pth`), Python cache, OS files, and the OCR temp file (`_plate_temp.jpg`) |

---

## 9. Summary of the EasyOCR → Custom CNN Transition

| Aspect | EasyOCR (Generation 1) | Custom CNN (Generation 2) |
|---|---|---|
| Scope of training data | General-purpose, multi-domain Devanagari text | Nepali plate character crops specifically |
| Task framing | Full-image OCR (detection + recognition combined) | Single-character classification (detection handled separately by YOLOv8) |
| Robustness to glare/angle/wear | Poor — not tuned for plate-specific conditions | Strong — trained on real plate crops under varied conditions |
| Language generality | Broad (supports many languages/scripts) — a source of imprecision here | Narrow, closed-set (only the exact characters used on Nepali plates) |
| Models required | 1 (plate detector) | 3 (plate detector, character detector, character classifier) |
| Failure mode | Single OCR misread can shift/corrupt the entire formatted string | Failures are isolated to individual character predictions, easier to trace |

The move from EasyOCR to a custom CNN was driven by a straightforward observation: EasyOCR's underlying model is generalized to handle many languages and text styles, and that generality directly worked against precision on a narrow, high-stakes task like plate character recognition. Training a small, purpose-built CNN exclusively on real Nepali plate character crops — paired with a dedicated character-segmentation model to isolate those crops in the first place — produced a system that generalizes far better across the lighting, angle, and wear conditions found in real-world vehicle photographs.

"""
Core LPR pipeline logic — no Streamlit dependency, so it can be reused by the
app, the data-collection script, and any batch/evaluation scripts.

Pipeline:
    1) Plate detection (YOLOv8)         -> crop the plate out of the full image
    2) Plate crop enhancement            -> padding, upscale, CLAHE, sharpen
    3) Character segmentation (YOLOv8)  -> bounding boxes of individual characters
    4) Character classification (CharCNN) -> read each character crop
    5) Assemble + format the plate string
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

IMG_SIZE_DEFAULT = 64
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
DEVANAGARI_DIGITS = set("०१२३४५६७८९")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Model definition — must match the training script exactly
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Model loaders (plain — no caching; wrap with st.cache_resource in the app)
# --------------------------------------------------------------------------
def load_plate_model(path):
    return YOLO(path)


def load_char_seg_model(path):
    return YOLO(path)


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


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------
def pad_box(box, image_shape, pad_ratio=0.08):
    """Expands a box by pad_ratio on each side, clipped to image bounds.

    Tight YOLO boxes often clip the first/last character on a plate, which
    then produces a wrong reading downstream — a small margin fixes that.
    """
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
    """Returns the cropped plate (BGR ndarray) with the highest-confidence box, or None."""
    results = plate_model.predict(source=image_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    # pick the most confident detection if several are found
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])
    x1, y1, x2, y2 = pad_box((x1, y1, x2, y2), image_bgr.shape, pad_ratio=pad_ratio)
    crop = image_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def enhance_plate_crop(crop_bgr, target_min_height=180, max_scale=6.0,
                        apply_clahe=True, apply_sharpen=True):
    """
    Improves a low-quality plate crop before it goes into character segmentation.

    Deliberately stays a normal color image (no grayscale/thresholding) — the
    CNN classifier expects color input normalized like its training data, so
    binarizing here would push inputs off-distribution and hurt accuracy.
    """
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


def order_character_boxes(xyxy):
    """
    Returns box indices in reading order: top row(s) first, left-to-right within each row.

    Plain x-sorting breaks the moment a plate has more than one text line (very
    common on Nepali plates — province name above the plate number) or is even
    slightly rotated, since box x-ranges can then overlap across rows. This
    clusters boxes into rows by y-position first, then sorts each row by x.
    """
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


def segment_characters(char_seg_model, plate_bgr, conf):
    """Returns a list of character crops (BGR ndarray), in reading order (row-aware)."""
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


def classify_characters(cnn_model, class_names, transform, char_crops):
    """Returns a list of predicted character strings, one per crop."""
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


def format_plate(chars):
    """
    Formats the recognized sequence as Nepali plates are actually laid out:
    [[prefix: province char(s) + zone digit(s) + class letter]] [4-digit number].

    Only the trailing 4-digit number is a fixed-length part of a Nepali plate —
    the zone-digit count (1 or 2 digits depending on province) and the number
    of prefix letters both vary. Assuming a fixed zone-digit count caused
    digits to be double-counted (once as "zone", again as part of the number)
    whenever a plate's actual zone code was shorter than assumed. So this just
    inserts a space before the last 4 digits and leaves everything else as-is,
    rather than re-deriving groups from scratch.
    """
    raw = "".join(chars)

    if len(chars) >= 4 and all(c in DEVANAGARI_DIGITS for c in chars[-4:]):
        prefix = "".join(chars[:-4])
        number = "".join(chars[-4:])
        return f"{prefix} {number}", raw

    return raw, raw
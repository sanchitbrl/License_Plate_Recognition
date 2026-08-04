# Nepali License Plate Recognition (LPR)

An end-to-end pipeline for detecting and reading Nepali vehicle license plates from images. Given a photo of a vehicle, the system localizes the plate region, enhances it, extracts the alphanumeric (Devanagari) characters, and reconstructs the plate string in the standard Nepali format.

The repository originated as two Colab notebooks (`LPR.py` and `character_detection_model.py`) used for dataset preparation and model training/evaluation, and has since evolved into two deployable Streamlit applications implementing distinct recognition strategies.

## Pipeline overview

At a high level, both apps follow the same four-stage pipeline:

1. **Plate detection** — A YOLOv8 object detection model localizes the plate's bounding box within the full vehicle image.
2. **Crop enhancement** — The detected region is cropped and preprocessed (upscaling, contrast/sharpness adjustments) to improve legibility before recognition.
3. **Character recognition** — The two apps diverge here; see below for implementation details.
4. **Plate formatting** — Recognized characters are reassembled into the standard Nepali plate layout, e.g. `बा२ख 1234`.

## Two recognition strategies

The repo implements two distinct approaches to the character-recognition stage, each exposed as a separate Streamlit application. `app.py` (EasyOCR) was the initial implementation; `app2.py` (custom CNN) was developed afterward once EasyOCR's accuracy proved insufficient for this use case, and is the recommended approach going forward.

### `app.py` — YOLOv8 + EasyOCR

A lighter-weight, OCR-based approach:
- Runs a YOLOv8 model to detect and crop the plate region.
- Applies classical preprocessing to the crop (grayscale conversion, Gaussian blur, binary inverse thresholding, and a sharpening convolution kernel).
- Passes the processed crop to **EasyOCR**, configured with the Nepali (`ne`) language pack and a character allowlist restricted to valid Devanagari digits and plate letters.
- Post-processes the raw OCR output by separating recognized characters into letter and digit sets, then reconstructs the plate string according to the expected `[char][2-digit zone][char] [4-digit number]` layout.

This approach only requires a single trained artifact (the plate detector), since character recognition is delegated to EasyOCR's pretrained model. It's faster to stand up, but in practice its accuracy was not sufficient for reliable production use — since EasyOCR is a general-purpose OCR engine and not fine-tuned on Nepali plate characters, it's noticeably sensitive to font irregularities, glare, viewing angle, and plate wear. This limitation was the direct motivation for the second approach below.

### `app2.py` — YOLOv8 + Character Segmentation + Custom CNN

A more involved, fully custom pipeline, with core logic factored out into `pipeline.py` for reuse and testability:
- Detects and pads the plate bounding box (padding compensates for tight YOLO boxes that tend to clip edge characters).
- Enhances the crop via upscaling, CLAHE (contrast-limited adaptive histogram equalization), and unsharp masking — deliberately kept in color rather than binarized, since the downstream CNN classifier is trained on color inputs and normalized accordingly.
- Runs a **second YOLOv8 model** trained specifically for individual character detection on the plate crop.
- Determines reading order via row-aware clustering of bounding boxes (grouped by vertical center, then sorted left-to-right within each row) rather than a naive x-coordinate sort — necessary because Nepali plates frequently span two text rows (e.g., province designation above the plate number) and can be slightly rotated.
- Classifies each segmented character crop using a custom-trained CNN (`CharCNN`, defined in `pipeline.py`) — a compact convolutional architecture (three conv blocks with batch norm and max pooling, followed by a fully connected classifier head).
- Formats the final string by identifying the trailing 4-digit numeric sequence and treating everything preceding it as the prefix, rather than assuming fixed-length groupings for the zone code (which varies by province and previously caused digit misalignment).

This pipeline requires three trained artifacts instead of one (plate detector, character segmenter, character classifier), but was built specifically to address the accuracy shortfall of the EasyOCR approach. Because every stage — detection, segmentation, and classification — is trained end-to-end on domain-specific data rather than relying on a general-purpose OCR engine, it achieves substantially better accuracy and generalizes well across varied real-world conditions (lighting, angle, plate wear, and font variation). This is the approach recommended for production use.

## Project structure

```
.
├── app.py                        # Streamlit app — YOLO + EasyOCR approach
├── app2.py                       # Streamlit app — YOLO + character CNN approach
├── pipeline.py                   # Core logic used by app2.py (detection, enhancement, segmentation, classification, formatting)
├── LPR.py                        # Original Colab notebook — plate detection + OCR experiments
├── character_detection_model.py  # Original Colab notebook — character segmentation model training
├── README.md
└── requirements.txt
```

Trained model weights (`.pt`, `.pth`) are not tracked in version control (see `.gitignore`) and must be supplied separately.

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Provide the model weights.** These are trained separately — refer to `LPR.py` and `character_detection_model.py` for the original training procedure — and are not included in this repository.

   - For `app.py`:
     - `best(LP dectection).pt` — the plate detector, expected in the project root.
   - For `app2.py`, three files inside a `models/` directory next to `app2.py`:
     - `models/plate_detector.pt`
     - `models/char_segmenter.pt`
     - `models/char_cnn_checkpoint.pth`

   `app2.py` performs a startup check and will report any missing files explicitly rather than failing silently.

## Running the apps

Launch either app via Streamlit:

```bash
streamlit run app.py
```

or

```bash
streamlit run app2.py
```

Open the local URL Streamlit provides, upload a vehicle image, and the pipeline will execute end-to-end, displaying intermediate outputs (detection box, crop, preprocessing/enhancement result, and segmented characters where applicable) alongside the final formatted plate string.

## Plate format reference

Nepali plates generally follow this layout:

```
[Province letter(s)] [Zone digit(s)] [Vehicle class letter]   [4-digit number]
```

Only the trailing 4-digit number is a fixed-length, reliably identifiable component — the zone-digit count and prefix length vary by province. For this reason, `pipeline.py::format_plate` does not attempt to parse fixed-size groups; it identifies the trailing 4-digit sequence and treats everything preceding it as an opaque prefix. This avoids the digit-misalignment issue that occurs when a shorter-than-assumed zone code causes overlap with the number field.

## Known limitations

- No pretrained weights are bundled with this repository; both apps require external model artifacts to run.
- `app.py`'s OCR-based recognition is retained mainly for reference/comparison; its accuracy was found insufficient for reliable use and it is more susceptible to degradation from atypical fonts, glare, motion blur, or physical plate damage, given its reliance on a general-purpose OCR engine rather than a domain-specific classifier. `app2.py` is the recommended approach.
- Both apps currently process only the single highest-confidence plate detection per image; multi-vehicle images with multiple plates are not fully handled.
- The Colab notebooks (`LPR.py`, `character_detection_model.py`) are retained for reference and reproducibility and are not intended to run outside the Colab environment as-is, since they depend on `google.colab` upload/display utilities.
- `app.py` writes an intermediate file (`_plate_temp.jpg`) to disk during OCR; this is a side effect of EasyOCR's file-based `readtext` API and is excluded via `.gitignore`.


## Demo Video
[![Watch the video](https://www.youtube.com/watch?v=ILccmyscx8c)](https://www.youtube.com/watch?v=ILccmyscx8c)

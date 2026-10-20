"""
OCR Parser Module for Biosafety Document Ingestion.
Implements prioritized OCR extraction:
1. RapidOCR (Fast, high-accuracy Korean text detection via ONNX)
2. Tesseract OCR (with multi-stage preprocessing, contrast boost, adaptive thresholding)
3. Structured logging and failed page preservation in data/failed/.
"""
import os
import re
from typing import Dict, Any, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

from config import config
from pipeline.logger import logger, IngestionErrorType, log_pipeline_event, save_failed_page

# Try RapidOCR
try:
    from rapidocr import RapidOCR
    HAS_RAPIDOCR = True
    _rapidocr_engine = None
except ImportError:
    try:
        from rapidocr_onnxruntime import RapidOCR
        HAS_RAPIDOCR = True
        _rapidocr_engine = None
    except ImportError:
        HAS_RAPIDOCR = False
        _rapidocr_engine = None
        logger.warning("RapidOCR is not installed.")

# Try Tesseract
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False
    logger.warning("pytesseract is not installed.")


def get_tesseract_executable() -> Optional[str]:
    """Finds configured or detected Tesseract executable."""
    return config.get_tesseract_path()


def _get_rapidocr_instance():
    """Lazily initializes RapidOCR singleton."""
    global _rapidocr_engine
    if HAS_RAPIDOCR and _rapidocr_engine is None:
        try:
            _rapidocr_engine = RapidOCR()
            logger.info("Initialized RapidOCR engine.")
        except Exception as e:
            logger.error(f"Failed to initialize RapidOCR: {e}")
            _rapidocr_engine = None
    return _rapidocr_engine


def preprocess_image(image: Image.Image, mode: str = "standard") -> Image.Image:
    """
    Applies image enhancement and binarization for OCR accuracy.
    modes:
    - 'standard': Grayscale + contrast enhancement + mild sharpen
    - 'adaptive': Grayscale + high contrast + thresholding
    - 'invert_cleanup': Invert if background is dark, auto-contrast
    """
    # Ensure RGB / L mode
    if image.mode != "RGB" and image.mode != "L":
        img = image.convert("RGB")
    else:
        img = image.copy()

    # Grayscale conversion
    gray = ImageOps.grayscale(img)

    if mode == "standard":
        # Mild contrast enhancement and unsharp mask
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        return sharpened

    elif mode == "adaptive":
        # Higher contrast + autocontrast
        auto = ImageOps.autocontrast(gray, cutoff=2)
        enhancer = ImageEnhance.Contrast(auto)
        enhanced = enhancer.enhance(2.2)
        # Apply simple thresholding
        np_img = np.array(enhanced)
        mean_val = np.mean(np_img)
        thresh_val = max(100, min(160, int(mean_val * 0.9)))
        binarized = (np_img > thresh_val).astype(np.uint8) * 255
        return Image.fromarray(binarized)

    elif mode == "invert_cleanup":
        auto = ImageOps.autocontrast(gray)
        return auto

    return gray


def postprocess_ocr_text(raw_text: str) -> str:
    """
    Cleans OCR output while preserving legal articles and tabular structures.
    Preserves:
    - Legal article markers: '제1조', '제2조의2', '제3조(정의)'
    - List items: '1.', '가.', '(1)', '①'
    - Table dividers: '|'
    Removes:
    - Broken special character glitches
    - Excessive spaces between Korean and numbers
    - Unnecessary mid-sentence line breaks
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Normalize unicode characters (NFKC)
    import unicodedata
    text = unicodedata.normalize("NFKC", text)

    # 2. Fix multiple redundant whitespace inside lines
    text = re.sub(r"[ \t]+", " ", text)

    # 3. Clean up broken OCR character glitches (e.g. repeated lone punctuation '`', '~', '^')
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"(?<=\s)[`'\"^~](?=\s)", "", text)

    # 4. Collapse spaces between Korean characters and adjacent punctuation/digits
    # e.g., "제 1 조" -> "제1조"
    text = re.sub(r"제\s+(\d+)\s+조", r"제\1조", text)
    text = re.sub(r"제\s+(\d+)\s+항", r"제\1항", text)

    # 5. Sentence line wrap unwrapping:
    # If a line does not end with a sentence terminator (. ? ! : ; 조 항 호 목 [표 시작] [표 종료])
    # and the next line does not start with an item marker, merge lines.
    lines = text.split("\n")
    merged_lines = []
    buffer = ""

    item_marker_regex = re.compile(r"^(\d+\.|\([0-9가-힣]+\)|[가-힣]\.|[①-⑳]|제\d+조|\[표)")

    for line in lines:
        line_str = line.strip()
        if not line_str:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            continue

        if not buffer:
            buffer = line_str
        else:
            # Check if previous buffer looks like an incomplete clause
            last_char = buffer[-1]
            is_item_start = bool(item_marker_regex.match(line_str))
            is_table_line = "|" in buffer or "|" in line_str

            if (last_char not in (".", "!", "?", ":", "]", ")", ">")
                    and not is_item_start
                    and not is_table_line):
                # Join with a single space
                buffer = f"{buffer} {line_str}"
            else:
                merged_lines.append(buffer)
                buffer = line_str

    if buffer:
        merged_lines.append(buffer)

    cleaned = "\n".join(merged_lines)
    return cleaned.strip()


def extract_with_rapidocr(image: Image.Image) -> Dict[str, Any]:
    """Runs RapidOCR on image and calculates average confidence score."""
    engine = _get_rapidocr_instance()
    if engine is None:
        raise RuntimeError("RapidOCR engine could not be loaded.")

    img_np = np.array(image.convert("RGB"))
    output = engine(img_np)

    raw_lines = []
    scores = []

    # Handle RapidOCROutput object (RapidOCR 3.x+)
    if hasattr(output, "txts") and output.txts:
        raw_lines = [str(t).strip() for t in output.txts if str(t).strip()]
        if hasattr(output, "scores") and output.scores:
            scores = [float(s) * 100.0 for s in output.scores]
    # Handle legacy tuple (results, elapse)
    elif isinstance(output, (tuple, list)) and len(output) > 0 and output[0]:
        for item in output[0]:
            if len(item) >= 3:
                text = str(item[1]).strip()
                score = float(item[2]) * 100.0
                if text:
                    raw_lines.append(text)
                    scores.append(score)

    if not raw_lines:
        return {
            "text": "",
            "raw_text": "",
            "confidence": 0.0,
            "engine": "rapidocr",
            "status": "empty"
        }

    raw_text = "\n".join(raw_lines)
    avg_conf = round(float(np.mean(scores)), 2) if scores else 0.0
    cleaned_text = postprocess_ocr_text(raw_text)

    return {
        "text": cleaned_text,
        "raw_text": raw_text,
        "confidence": avg_conf,
        "engine": "rapidocr",
        "status": "success"
    }


def extract_with_tesseract(
    image: Image.Image,
    language: str = "kor+eng",
    preprocess_mode: str = "standard"
) -> Dict[str, Any]:
    """Runs Tesseract OCR with detailed confidence metrics."""
    if not HAS_PYTESSERACT:
        raise RuntimeError("pytesseract is not installed.")

    tess_path = get_tesseract_executable()
    if not tess_path:
        raise FileNotFoundError(
            "Tesseract executable not found. Please install Tesseract-OCR and ensure it is in PATH "
            "or set TESSERACT_CMD in .env (e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)."
        )

    pytesseract.pytesseract.tesseract_cmd = tess_path

    processed_img = preprocess_image(image, mode=preprocess_mode)

    # Use image_to_data to extract confidence scores
    try:
        data = pytesseract.image_to_data(
            processed_img,
            lang=language,
            output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        # Fallback to kor alone if kor+eng failed
        if "+" in language:
            data = pytesseract.image_to_data(
                processed_img,
                lang="kor",
                output_type=pytesseract.Output.DICT
            )
        else:
            raise e

    # Extract words and confidences
    n_boxes = len(data["text"])
    words = []
    confidences = []
    current_line = []
    last_line_num = -1

    for i in range(n_boxes):
        w = data["text"][i].strip()
        conf = float(data["conf"][i])
        line_num = data["line_num"][i]

        if conf > 0 and w:
            confidences.append(conf)
            if last_line_num != -1 and line_num != last_line_num:
                words.append(" ".join(current_line))
                current_line = [w]
            else:
                current_line.append(w)
            last_line_num = line_num

    if current_line:
        words.append(" ".join(current_line))

    raw_text = "\n".join(words)
    avg_conf = round(float(np.mean(confidences)), 2) if confidences else 0.0
    cleaned_text = postprocess_ocr_text(raw_text)

    return {
        "text": cleaned_text,
        "raw_text": raw_text,
        "confidence": avg_conf,
        "engine": "tesseract",
        "status": "success" if cleaned_text else "empty"
    }


def extract_text_with_ocr(
    image: Union[Image.Image, np.ndarray, str],
    language: str = "kor+eng",
    dpi: int = 300,
    preprocess: bool = True
) -> Dict[str, Any]:
    """
    Standard OCR interface:
    Extracts text with prioritized engine selection:
    1. RapidOCR (if preferred and available)
    2. Tesseract OCR (with confidence checking and multi-mode retry)
    3. Error capture and diagnostic reporting

    Returns:
    {
        "text": "...",
        "raw_text": "...",
        "confidence": 81.2,
        "engine": "rapidocr" | "tesseract",
        "status": "success" | "failed" | "empty"
    }
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # 1. Try RapidOCR first if available and preferred
    if config.prefer_rapidocr and HAS_RAPIDOCR:
        try:
            res = extract_with_rapidocr(image)
            if res["status"] == "success" and res["confidence"] >= config.ocr_min_confidence:
                return res
            logger.info(f"RapidOCR confidence ({res['confidence']}) below threshold or empty. Checking fallback.")
        except Exception as e:
            logger.warning(f"RapidOCR failed: {e}. Falling back to Tesseract.")

    # 2. Try Tesseract OCR
    if HAS_PYTESSERACT and get_tesseract_executable():
        modes_to_try = ["standard", "adaptive"] if preprocess else ["none"]
        best_result = None

        for mode in modes_to_try:
            try:
                res = extract_with_tesseract(
                    image,
                    language=language or config.ocr_language,
                    preprocess_mode=mode
                )
                if best_result is None or res["confidence"] > best_result["confidence"]:
                    best_result = res

                if res["confidence"] >= config.ocr_min_confidence and res["text"]:
                    return res
            except Exception as e:
                logger.warning(f"Tesseract OCR mode '{mode}' failed: {e}")

        if best_result and best_result["text"]:
            return best_result

    # 3. If RapidOCR was not tried yet and Tesseract failed, try RapidOCR now
    if HAS_RAPIDOCR:
        try:
            res = extract_with_rapidocr(image)
            if res["text"]:
                return res
        except Exception as e:
            logger.error(f"Fallback RapidOCR failed: {e}")

    # 4. Total Failure
    return {
        "text": "",
        "raw_text": "",
        "confidence": 0.0,
        "engine": "none",
        "status": "failed",
        "error": "All OCR engines failed or are unavailable."
    }

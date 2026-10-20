"""
PDF Classification and State Analysis Module.
Analyzes PDF structure using PyMuPDF (fitz) to identify:
Type A: Normal text layer
Type B: Mixed (partial text layer, partial scan)
Type C: Fully scanned image PDF
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import config
from pipeline.logger import logger, IngestionErrorType, log_pipeline_event

try:
    import pymupdf as fitz
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False
        logger.warning("PyMuPDF is not available. Will fallback to pypdf for classification.")


def classify_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Examines each page's text character count to classify the PDF state.

    Returns:
    {
        "document_path": str,
        "page_count": int,
        "text_page_count": int,
        "image_page_count": int,
        "is_scanned": bool,
        "is_mixed": bool,
        "average_chars_per_page": float,
        "pages": List[Dict[str, Any]]
    }
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    min_chars = config.min_text_chars_per_page
    scanned_ratio_threshold = config.scanned_page_ratio

    pages_info: List[Dict[str, Any]] = []
    total_chars = 0

    if HAS_PYMUPDF:
        try:
            doc = fitz.open(str(file_path))
            page_count = len(doc)

            for page_index in range(page_count):
                page = doc[page_index]
                text = page.get_text() or ""
                text_clean = text.strip()
                char_count = len(text_clean)
                total_chars += char_count

                # Has images?
                image_list = page.get_images(full=True)
                has_images = len(image_list) > 0

                is_page_scanned = char_count < min_chars

                pages_info.append({
                    "page": page_index + 1,
                    "char_count": char_count,
                    "has_images": has_images,
                    "image_count": len(image_list),
                    "is_scanned": is_page_scanned,
                    "needs_ocr": is_page_scanned
                })
            doc.close()
        except Exception as e:
            log_pipeline_event(
                filename=file_path.name,
                parser="fitz_classifier",
                error_type=IngestionErrorType.PDF_PARSE_ERROR,
                message=f"Error inspecting PDF with PyMuPDF: {e}",
                status="failed"
            )
            # Fallback to pypdf
            return _classify_pdf_with_pypdf(file_path, min_chars, scanned_ratio_threshold)
    else:
        return _classify_pdf_with_pypdf(file_path, min_chars, scanned_ratio_threshold)

    return _summarize_classification(file_path, pages_info, total_chars, scanned_ratio_threshold)


def _classify_pdf_with_pypdf(file_path: Path, min_chars: int, scanned_ratio_threshold: float) -> Dict[str, Any]:
    """Fallback classifier using pypdf."""
    import pypdf
    pages_info = []
    total_chars = 0
    reader = pypdf.PdfReader(str(file_path))
    page_count = len(reader.pages)

    for idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        char_count = len(text)
        total_chars += char_count
        is_page_scanned = char_count < min_chars
        pages_info.append({
            "page": idx + 1,
            "char_count": char_count,
            "has_images": is_page_scanned,
            "image_count": 1 if is_page_scanned else 0,
            "is_scanned": is_page_scanned,
            "needs_ocr": is_page_scanned
        })

    return _summarize_classification(file_path, pages_info, total_chars, scanned_ratio_threshold)


def _summarize_classification(
    file_path: Path,
    pages_info: List[Dict[str, Any]],
    total_chars: int,
    scanned_ratio_threshold: float
) -> Dict[str, Any]:
    page_count = len(pages_info)
    if page_count == 0:
        return {
            "document_path": str(file_path),
            "page_count": 0,
            "text_page_count": 0,
            "image_page_count": 0,
            "is_scanned": False,
            "is_mixed": False,
            "average_chars_per_page": 0.0,
            "pages": []
        }

    image_page_count = sum(1 for p in pages_info if p["is_scanned"])
    text_page_count = page_count - image_page_count
    scanned_ratio = image_page_count / page_count
    avg_chars = round(total_chars / page_count, 2)

    is_scanned = scanned_ratio >= scanned_ratio_threshold
    is_mixed = 0 < image_page_count < page_count

    return {
        "document_path": str(file_path.resolve()),
        "page_count": page_count,
        "text_page_count": text_page_count,
        "image_page_count": image_page_count,
        "is_scanned": is_scanned,
        "is_mixed": is_mixed,
        "average_chars_per_page": avg_chars,
        "pages": pages_info
    }

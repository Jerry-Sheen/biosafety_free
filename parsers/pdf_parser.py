"""
Specialized PDF Parser with Hybrid Native Text, Table Extraction, and Adaptive OCR.
Coordinates PyMuPDF (fitz), Docling, and OCR to preserve document structure,
tables, and page metadata without redundant duplicates.
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import io

from config import config
from parsers import DocumentRecord
from parsers.ocr_parser import extract_text_with_ocr
from pipeline.classify import classify_pdf
from pipeline.quality import evaluate_text_quality
from pipeline.logger import (
    logger,
    IngestionErrorType,
    log_pipeline_event,
    save_failed_page
)

try:
    import pymupdf as fitz
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False

# Docling availability check flag
HAS_DOCLING = None

def _check_docling():
    global HAS_DOCLING
    if HAS_DOCLING is None:
        try:
            import docling
            HAS_DOCLING = True
        except ImportError:
            HAS_DOCLING = False
    return HAS_DOCLING


class PDFParser:
    """Intelligent PDF Parser with table preservation, page tracking, and adaptive OCR."""

    def __init__(self, enable_docling: bool = True):
        self.enable_docling = enable_docling
        self._docling_converter = None

    def _get_docling_converter(self):
        if self._docling_converter is None and self.enable_docling and _check_docling():
            try:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.base_models import InputFormat

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False  # Keep digital Docling light; we do targeted OCR
                pipeline_options.do_table_structure = True
                format_options = {
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
                self._docling_converter = DocumentConverter(format_options=format_options)
            except Exception as e:
                logger.warning(f"Docling initialization failed: {e}. Falling back to PyMuPDF native extraction.")
                self.enable_docling = False
        return self._docling_converter

    def parse(self, file_path: Path) -> List[DocumentRecord]:
        """
        Parses PDF according to prioritized strategy:
        1. Classify PDF & inspect pages
        2. Extract native text & tables (PyMuPDF / Docling)
        3. For scanned/low-quality pages, perform 300 DPI OCR
        4. Select highest quality representation per page (avoid duplicate concatenation)
        5. Separate tables into distinct DocumentRecords
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Compute file hash
        file_bytes = file_path.read_bytes()
        doc_hash = hashlib.sha256(file_bytes).hexdigest()

        # Step 1: Classify PDF
        classification = classify_pdf(file_path)
        logger.info(
            f"Classified PDF {file_path.name}: {classification['page_count']} pages "
            f"(text pages: {classification['text_page_count']}, image/scanned: {classification['image_page_count']}, "
            f"avg_chars: {classification['average_chars_per_page']})"
        )

        records: List[DocumentRecord] = []

        if HAS_PYMUPDF:
            records = self._parse_with_pymupdf(file_path, classification, doc_hash)
        else:
            records = self._parse_fallback(file_path, classification, doc_hash)

        return records

    def _parse_with_pymupdf(
        self,
        file_path: Path,
        classification: Dict[str, Any],
        doc_hash: str
    ) -> List[DocumentRecord]:
        doc = fitz.open(str(file_path))
        all_records: List[DocumentRecord] = []
        source_name = file_path.name
        source_path = str(file_path.resolve())

        for page_index in range(len(doc)):
            page_num = page_index + 1
            page = doc[page_index]
            page_meta_base = {
                "source": source_name,
                "source_path": source_path,
                "page": page_num,
                "total_pages": len(doc),
                "document_hash": doc_hash
            }

            # 1. Extract Native Tables
            extracted_tables: List[Dict[str, Any]] = []
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    for t_idx, table in enumerate(tables.tables):
                        df = table.to_pandas()
                        if not df.empty:
                            md_table = df.to_markdown(index=False)
                            rows, cols = df.shape
                            table_id = f"page_{page_num}_table_{t_idx + 1}"
                            table_text = f"[표 시작]\n{md_table}\n[표 종료]"
                            table_meta = dict(page_meta_base)
                            table_meta.update({
                                "content_type": "table",
                                "table_id": table_id,
                                "rows": rows,
                                "columns": cols,
                                "extractor": "pymupdf_table",
                                "table_quality": "high" if cols > 1 and rows > 0 else "low"
                            })
                            extracted_tables.append({
                                "record": DocumentRecord(
                                    text=table_text,
                                    raw_text=md_table,
                                    metadata=table_meta
                                ),
                                "bbox": table.bbox
                            })
            except Exception as e:
                logger.warning(f"Table extraction warning on page {page_num}: {e}")

            # 2. Extract Native Text
            native_text = (page.get_text() or "").strip()
            native_quality = evaluate_text_quality(native_text)

            chosen_text = native_text
            extractor = "pymupdf_native"
            ocr_used = False
            ocr_engine = "none"
            ocr_confidence = 0.0

            # 3. Decide if OCR is necessary
            needs_ocr = (
                native_quality["needs_retry"]
                or len(native_text) < config.min_text_chars_per_page
            )

            ocr_image = None
            if needs_ocr:
                logger.info(f"Page {page_num} requires OCR (chars: {len(native_text)}, quality: {native_quality['quality_score']:.2f})")
                try:
                    # Render page at 300 DPI
                    pix = page.get_pixmap(dpi=config.ocr_dpi)
                    img_data = pix.tobytes("png")
                    ocr_image = Image.open(io.BytesIO(img_data))

                    ocr_res = extract_text_with_ocr(
                        ocr_image,
                        language=config.ocr_language,
                        dpi=config.ocr_dpi,
                        preprocess=True
                    )

                    ocr_text = ocr_res.get("text", "").strip()
                    ocr_conf = float(ocr_res.get("confidence", 0.0))
                    ocr_quality = evaluate_text_quality(ocr_text, ocr_confidence=ocr_conf)

                    # 4. Compare quality: Choose the superior result (Do NOT blindly concatenate!)
                    if ocr_quality["quality_score"] > native_quality["quality_score"] and ocr_text:
                        chosen_text = ocr_text
                        extractor = f"ocr_{ocr_res.get('engine', 'unknown')}"
                        ocr_used = True
                        ocr_engine = ocr_res.get("engine", "unknown")
                        ocr_confidence = ocr_conf
                    else:
                        # Keep native text if OCR didn't improve it
                        if len(ocr_text) > len(native_text) * 1.5 and ocr_conf >= config.ocr_min_confidence:
                            chosen_text = ocr_text
                            extractor = f"ocr_{ocr_res.get('engine', 'unknown')}"
                            ocr_used = True
                            ocr_engine = ocr_res.get("engine", "unknown")
                            ocr_confidence = ocr_conf

                    # If still failed, log diagnostics
                    if not chosen_text or (ocr_used and ocr_conf < config.ocr_min_confidence):
                        save_failed_page(
                            filename=source_name,
                            page_num=page_num,
                            image=ocr_image,
                            diagnostic_info={
                                "error_type": IngestionErrorType.OCR_LOW_CONFIDENCE,
                                "ocr_engine": ocr_engine,
                                "ocr_confidence": ocr_confidence,
                                "native_quality": native_quality,
                                "ocr_quality": ocr_quality
                            },
                            extracted_text=chosen_text
                        )
                except Exception as e:
                    log_pipeline_event(
                        filename=source_name,
                        page=page_num,
                        parser="ocr_processor",
                        error_type=IngestionErrorType.OCR_INSTALLATION_ERROR if "Tesseract" in str(e) else IngestionErrorType.UNKNOWN_ERROR,
                        message=f"OCR execution failed on page {page_num}: {e}",
                        status="failed"
                    )
                    save_failed_page(
                        filename=source_name,
                        page_num=page_num,
                        image=ocr_image,
                        diagnostic_info={"error": str(e)},
                        extracted_text=native_text
                    )

            # 5. Create Body Record
            if chosen_text:
                body_meta = dict(page_meta_base)
                body_meta.update({
                    "content_type": "body",
                    "extractor": extractor,
                    "ocr_used": ocr_used,
                    "ocr_engine": ocr_engine,
                    "ocr_confidence": ocr_confidence,
                    "quality_score": native_quality["quality_score"] if not ocr_used else ocr_confidence / 100.0
                })
                all_records.append(DocumentRecord(
                    text=chosen_text,
                    raw_text=native_text,
                    metadata=body_meta
                ))

            # 6. Add Extracted Table Records
            for t_item in extracted_tables:
                all_records.append(t_item["record"])

        doc.close()
        return all_records

    def _parse_fallback(
        self,
        file_path: Path,
        classification: Dict[str, Any],
        doc_hash: str
    ) -> List[DocumentRecord]:
        """Fallback when PyMuPDF is not installed."""
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        records: List[DocumentRecord] = []
        source_name = file_path.name
        source_path = str(file_path.resolve())

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            text = (page.extract_text() or "").strip()
            meta = {
                "source": source_name,
                "source_path": source_path,
                "page": page_num,
                "total_pages": len(reader.pages),
                "document_hash": doc_hash,
                "content_type": "body",
                "extractor": "pypdf_fallback",
                "ocr_used": False,
                "ocr_engine": "none",
                "ocr_confidence": 0.0
            }
            if text:
                records.append(DocumentRecord(text=text, raw_text=text, metadata=meta))

        return records

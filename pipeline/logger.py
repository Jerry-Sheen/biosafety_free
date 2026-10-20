"""
Logging and Error Categorization for Biosafety RAG Pipeline.
Writes formatted logs to logs/ingestion.log and stores failed page diagnostics in data/failed/.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from config import config

# Standard Error Categories
class IngestionErrorType:
    FILE_READ_ERROR = "FILE_READ_ERROR"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    PDF_PARSE_ERROR = "PDF_PARSE_ERROR"
    OCR_INSTALLATION_ERROR = "OCR_INSTALLATION_ERROR"
    OCR_LOW_CONFIDENCE = "OCR_LOW_CONFIDENCE"
    TABLE_EXTRACTION_ERROR = "TABLE_EXTRACTION_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    VECTORSTORE_ERROR = "VECTORSTORE_ERROR"
    GEMINI_MODEL_NOT_FOUND = "GEMINI_MODEL_NOT_FOUND"
    GEMINI_AUTH_ERROR = "GEMINI_AUTH_ERROR"
    GEMINI_RATE_LIMIT = "GEMINI_RATE_LIMIT"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


def setup_ingestion_logger() -> logging.Logger:
    """Configures structured logger writing to both console and logs/ingestion.log."""
    logger = logging.getLogger("ingestion_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        config.init_directories()
        log_path = config.log_file

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File Handler
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


logger = setup_ingestion_logger()


def log_pipeline_event(
    filename: str,
    page: Optional[int] = None,
    parser: str = "unknown",
    ocr_engine: Optional[str] = None,
    error_type: Optional[str] = None,
    message: str = "",
    retry_count: int = 0,
    status: str = "success",
    extra: Optional[Dict[str, Any]] = None
):
    """
    Logs structured audit events in adherence to specification.
    """
    timestamp = datetime.now().isoformat()
    record = {
        "timestamp": timestamp,
        "filename": filename,
        "page": page,
        "parser": parser,
        "ocr_engine": ocr_engine or "none",
        "error_type": error_type,
        "message": message,
        "retry_count": retry_count,
        "status": status,
        "extra": extra or {}
    }

    log_line = (
        f"file={filename} | page={page} | parser={parser} | ocr={ocr_engine} | "
        f"status={status} | retry={retry_count} | error={error_type} | msg={message}"
    )

    if status == "failed" or error_type:
        logger.error(log_line)
    elif status == "warning":
        logger.warning(log_line)
    else:
        logger.info(log_line)


def save_failed_page(
    filename: str,
    page_num: int,
    image = None,
    diagnostic_info: Optional[Dict[str, Any]] = None,
    extracted_text: str = ""
):
    """
    Saves failed page diagnostics in data/failed/:
    - document_page_012.png
    - document_page_012.json
    - document_page_012.txt
    """
    try:
        config.init_directories()
        base_name = Path(filename).stem
        safe_name = f"{base_name}_page_{page_num:03d}"
        failed_dir = config.data_failed_dir

        # 1. Save Image
        if image is not None:
            png_path = failed_dir / f"{safe_name}.png"
            if hasattr(image, "save"):
                image.save(str(png_path))
            elif isinstance(image, bytes):
                png_path.write_bytes(image)

        # 2. Save JSON Diagnostics
        json_path = failed_dir / f"{safe_name}.json"
        diagnostics = diagnostic_info or {}
        diagnostics.update({
            "filename": filename,
            "page_num": page_num,
            "timestamp": datetime.now().isoformat(),
            "failure_record": True
        })
        json_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

        # 3. Save TXT (whatever was extracted or raw)
        txt_path = failed_dir / f"{safe_name}.txt"
        txt_path.write_text(extracted_text or "[NO_TEXT_EXTRACTED]", encoding="utf-8")

        logger.warning(f"Saved failed page diagnostics to {failed_dir / safe_name}.*")
    except Exception as e:
        logger.error(f"Failed to record diagnostics for failed page {filename} p.{page_num}: {e}")

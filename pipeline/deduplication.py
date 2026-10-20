"""
Deduplication and End-to-End Document Ingestion Pipeline.
Handles:
1. File classification & format dispatching
2. Structure-preserving parsing & adaptive OCR
3. Quality checks & table segregation
4. Hash-based deduplication (document_hash & chunk_hash)
5. Vector store ingestion and structured audit reporting
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from config import config
from parsers import DocumentRecord
from parsers.pdf_parser import PDFParser
from parsers.hwp_parser import HWPParser
from parsers.csv_parser import CSVParser
from parsers.markdown_parser import MarkdownParser
from pipeline.chunking import StructurePreservingChunker
from pipeline.logger import (
    logger,
    IngestionErrorType,
    log_pipeline_event
)
from vectorstore.store import VectorStore

SUPPORTED_EXTENSIONS = {".pdf", ".hwp", ".hwpx", ".csv", ".md", ".txt"}


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 hash of entire file binary content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def dispatch_parser(file_path: Path) -> List[DocumentRecord]:
    """Dispatches appropriate parser based on file extension."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        parser = PDFParser(enable_docling=True)
        return parser.parse(file_path)
    elif ext in (".hwp", ".hwpx"):
        return HWPParser.parse(file_path)
    elif ext == ".csv":
        return CSVParser.parse(file_path)
    elif ext in (".md", ".markdown"):
        return MarkdownParser.parse(file_path)
    elif ext == ".txt":
        # Simple text parser
        text = file_path.read_text(encoding="utf-8", errors="replace")
        doc_hash = compute_file_hash(file_path)
        meta = {
            "source": file_path.name,
            "source_path": str(file_path.resolve()),
            "page": 1,
            "content_type": "body",
            "extractor": "text_plain",
            "document_hash": doc_hash
        }
        return [DocumentRecord(text=text, raw_text=text, metadata=meta)]
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")


def ingest_document(
    file_path: Union[str, Path],
    force: bool = False,
    vector_store: Optional[VectorStore] = None
) -> Dict[str, Any]:
    """
    Ingests a single document into the RAG pipeline with deduplication and quality safeguards.

    Returns:
    {
        "source": str,
        "status": "success" | "skipped" | "failed",
        "pages": int,
        "chunks": int,
        "ocr_pages": int,
        "failed_pages": List[int],
        "warnings": List[str]
    }
    """
    file_path = Path(file_path)
    source_name = file_path.name
    warnings: List[str] = []
    failed_pages: List[int] = []

    if not file_path.exists():
        log_pipeline_event(
            filename=source_name,
            error_type=IngestionErrorType.FILE_READ_ERROR,
            message="File does not exist",
            status="failed"
        )
        return {
            "source": source_name,
            "status": "failed",
            "pages": 0,
            "chunks": 0,
            "ocr_pages": 0,
            "failed_pages": [],
            "warnings": [f"File not found: {file_path}"]
        }

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        log_pipeline_event(
            filename=source_name,
            error_type=IngestionErrorType.UNSUPPORTED_FORMAT,
            message=f"Unsupported format: {ext}",
            status="failed"
        )
        return {
            "source": source_name,
            "status": "failed",
            "pages": 0,
            "chunks": 0,
            "ocr_pages": 0,
            "failed_pages": [],
            "warnings": [f"Unsupported format {ext}"]
        }

    doc_hash = compute_file_hash(file_path)
    vstore = vector_store or VectorStore()

    # Check deduplication
    if not force and vstore.has_document(doc_hash):
        logger.info(f"Document '{source_name}' (hash: {doc_hash[:8]}...) already ingested. Skipping (use --force to overwrite).")
        return {
            "source": source_name,
            "status": "skipped",
            "pages": 0,
            "chunks": 0,
            "ocr_pages": 0,
            "failed_pages": [],
            "warnings": ["Document already exists in vector store. Skipped."]
        }

    # If force and exists, clear previous version of this document
    if force and vstore.has_document(doc_hash):
        logger.info(f"Force re-ingesting '{source_name}'. Deleting previous chunks...")
        vstore.delete_by_document_hash(doc_hash)

    # Step 1: Parse Document
    try:
        records = dispatch_parser(file_path)
    except Exception as e:
        log_pipeline_event(
            filename=source_name,
            error_type=IngestionErrorType.PDF_PARSE_ERROR if ext == ".pdf" else IngestionErrorType.FILE_READ_ERROR,
            message=f"Failed to parse document: {e}",
            status="failed"
        )
        return {
            "source": source_name,
            "status": "failed",
            "pages": 0,
            "chunks": 0,
            "ocr_pages": 0,
            "failed_pages": [1],
            "warnings": [str(e)]
        }

    if not records:
        warnings.append("Document produced 0 text/table records.")
        return {
            "source": source_name,
            "status": "failed",
            "pages": 0,
            "chunks": 0,
            "ocr_pages": 0,
            "failed_pages": [1],
            "warnings": warnings
        }

    # Calculate statistics
    pages_seen = set()
    ocr_pages_seen = set()

    for r in records:
        p = r.metadata.get("page", 1)
        pages_seen.add(p)
        if r.metadata.get("ocr_used"):
            ocr_pages_seen.add(p)

    # Step 2: Structure-Preserving Chunking
    chunker = StructurePreservingChunker()
    chunks = chunker.chunk_records(records)

    # Step 3: Vector Store Ingestion
    try:
        inserted = vstore.add_chunks(chunks, force=force)
    except Exception as e:
        log_pipeline_event(
            filename=source_name,
            error_type=IngestionErrorType.VECTORSTORE_ERROR,
            message=f"Error inserting into vector store: {e}",
            status="failed"
        )
        return {
            "source": source_name,
            "status": "failed",
            "pages": len(pages_seen),
            "chunks": 0,
            "ocr_pages": len(ocr_pages_seen),
            "failed_pages": list(pages_seen),
            "warnings": [f"Vector store insert error: {e}"]
        }

    log_pipeline_event(
        filename=source_name,
        parser=ext.lstrip("."),
        ocr_engine="adaptive" if ocr_pages_seen else "none",
        message=f"Ingested {len(pages_seen)} pages, {inserted} chunks",
        status="success"
    )

    return {
        "source": source_name,
        "status": "success",
        "pages": len(pages_seen),
        "chunks": inserted,
        "ocr_pages": len(ocr_pages_seen),
        "failed_pages": failed_pages,
        "warnings": warnings
    }


def ingest_directory(
    directory_path: Union[str, Path],
    recursive: bool = True,
    force: bool = False,
    vector_store: Optional[VectorStore] = None
) -> List[Dict[str, Any]]:
    """
    Ingests all supported documents found in directory.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    vstore = vector_store or VectorStore()
    results = []

    pattern = "**/*" if recursive else "*"
    all_files = [
        f for f in dir_path.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    from tqdm import tqdm
    logger.info(f"Found {len(all_files)} supported files in {dir_path}")

    for file_path in tqdm(all_files, desc="Ingesting Documents"):
        res = ingest_document(file_path, force=force, vector_store=vstore)
        results.append(res)

    return results


# Type alias helper
Union_Path = Any

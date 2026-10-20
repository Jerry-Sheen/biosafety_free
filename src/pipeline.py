# src/pipeline.py
"""Pipeline orchestrator for document processing.

This module defines a single function :func:`process_file` that runs the full
pipeline for a given file path and reports progress using a left‑to‑right
console progress bar (``tqdm``).  The pipeline consists of six logical stages:

1. **Ingestion / OCR** – :func:`src.data.ingest.ingest_document`
2. **Quality assessment** – :func:`src.parsers.quality.assess_quality`
3. **Chunking** – :func:`src.rag.chunker.chunk_document`
4. **Embedding** – placeholder (can be replaced with a real embedder)
5. **Vector‑store insertion** – :func:`src.rag.vector_store.add_documents`
6. **Optional Gemini summarisation** – :class:`src.rag.gemini_client.GeminiClient`

The progress bar updates after each stage, even if a later stage fails.  Any
exception is logged with the structured logger but does not prevent earlier
stages from completing.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from src.utils.structured_logger import get_structured_logger
from src.data.ingest import ingest_document
from src.parsers.quality import assess_quality
from src.rag.chunker import chunk_document
# Fast free embedding helper – uses models from ``src.embeddings.embedding_models``.
from src.embeddings.embedding_models import get_embedding_function

def embed_chunks(chunks: list[Any], model_name: str | None = None) -> list[Any]:
    """Embed a list of text chunks using a free model (default ``mini-l6``).

    Parameters
    ----------
    chunks: list[Any]
        List of text chunks (strings) produced by ``chunk_document``.
    model_name: str | None, optional
        Identifier of the embedding model to use (e.g. ``"bge-m3"`` or ``"mini-l6"``).
        If ``None`` the ``EMBEDDING_MODEL`` environment variable is consulted; the
        default fallback is ``"mini-l6"`` – a very fast, zero‑cost model.
    """
    # Normalise to strings.
    texts = [c if isinstance(c, str) else str(c) for c in chunks]
    embed_fn = get_embedding_function(model_name)
    return embed_fn(texts)

from src.rag.vector_store import add_documents
from src.rag.gemini_client import GeminiClient

logger = get_structured_logger(__name__)

def process_file(
    file_path: str | Path,
    force: bool = False,
    show_progress: bool = True,
    embed: bool = True,
    run_gemini: bool = True,
    embed_model: str | None = None,
) -> None:
    """Run the full processing pipeline for *file_path*.

    Parameters
    ----------
    file_path: str | Path
        Path to the document to be processed.
    force: bool, optional
        If ``True`` re‑process the file even when a duplicate hash is found.
    show_progress: bool, optional
        When ``True`` a ``tqdm`` progress bar with six steps is displayed.
    embed: bool, optional
        Whether to run the embedding step (stage 4).  Can be disabled for
        debugging.
    run_gemini: bool, optional
        Whether to invoke Gemini summarisation after storing the vectors.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error(
            "File not found",
            extra={"error_type": "FILE_READ_ERROR", "source": str(file_path)},
        )
        raise FileNotFoundError(f"{file_path} does not exist")

    total_stages = 6
    pbar: Optional[tqdm] = None
    if show_progress:
        pbar = tqdm(total=total_stages, desc="Processing", unit="stage")

    # Stage 1 – Ingestion / OCR
    try:
        record = ingest_document(file_path, force=force)
        logger.info("Ingestion completed", extra={"stage": 1, "source": str(file_path)})
    finally:
        if pbar:
            pbar.update(1)

    # Stage 2 – Quality assessment
    try:
        quality_metrics = assess_quality(record.text)
        logger.info(
            "Quality assessment completed",
            extra={"stage": 2, "metrics": quality_metrics},
        )
        if quality_metrics.get("needs_retry"):
            logger.warning(
                "Low quality detected – pipeline will continue but flag raised",
                extra={"stage": 2},
            )
    finally:
        if pbar:
            pbar.update(1)

    # Stage 3 – Chunking
    try:
        chunks = chunk_document(record.text)
        logger.info("Chunking completed", extra={"stage": 3, "chunk_count": len(chunks)})
    finally:
        if pbar:
            pbar.update(1)

    # Stage 4 – Embedding (optional)
    try:
        if embed:
            embedded_chunks = embed_chunks(chunks, model_name=embed_model)
        else:
            embedded_chunks = chunks
        logger.info("Embedding step completed", extra={"stage": 4})
    finally:
        if pbar:
            pbar.update(1)

    # Stage 5 – Vector‑store insertion
    try:
        add_documents(embedded_chunks)
        logger.info("Vector store insertion completed", extra={"stage": 5})
    finally:
        if pbar:
            pbar.update(1)

    # Stage 6 – Optional Gemini summarisation (non‑blocking)
    try:
        if run_gemini:
            gemini = GeminiClient()
            # Summarise the whole document – the Gemini client will handle model
            # selection and error categorisation internally.
            summary = gemini.summarise(record.text)
            logger.info(
                "Gemini summarisation completed",
                extra={"stage": 6, "summary": summary},
            )
        else:
            logger.info("Gemini step skipped", extra={"stage": 6})
    except Exception as exc:
        # Gemini failures must NOT abort the pipeline.
        logger.error(
            "Gemini summarisation failed",
            exc_info=exc,
            extra={"stage": 6, "error_type": "GEMINI_ERROR"},
        )
    finally:
        if pbar:
            pbar.update(1)
            pbar.close()

    logger.info("Processing pipeline finished", extra={"file": str(file_path)})

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process a single document with progress bar")
    parser.add_argument("path", type=str, help="Path to the document file")
    parser.add_argument("--force", action="store_true", help="Force re‑processing even if duplicate")
    args = parser.parse_args()
    process_file(args.path, force=args.force)

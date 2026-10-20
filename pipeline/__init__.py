"""
Pipeline package for biosafety document ingestion.
Provides classification, text quality scoring, structure-preserving chunking,
and end-to-end deduplicated document ingestion.
"""

def get_ingestion_functions():
    from pipeline.deduplication import ingest_document, ingest_directory
    return ingest_document, ingest_directory

__all__ = [
    "classify_pdf",
    "evaluate_text_quality",
    "StructurePreservingChunker"
]

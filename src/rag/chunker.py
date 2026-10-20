"""
Intelligent document chunker.
Splits structured document text and markdown into semantically coherent chunks.
Preserves Korean sentence boundaries, paragraphs, and markdown tables.
"""
import re
from typing import List, Dict, Any
from src.security.logger import get_secure_logger

logger = get_secure_logger("chunker")

# Regex to detect Korean sentence boundaries (. , ! ? followed by space or newline)
KOREAN_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?다요죠])\s+")


class DocumentChunker:
    """Chunks text while preserving context and Korean linguistic structures."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits full document text into overlapping chunks with rich metadata.
        Returns: list of dicts: {"text": str, "metadata": dict}
        """
        if not text or not text.strip():
            return []

        # Split initially by double newlines (paragraphs/markdown blocks)
        paragraphs = text.split("\n\n")
        raw_sections = []

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            if len(p_clean) <= self.chunk_size:
                raw_sections.append(p_clean)
            else:
                # If paragraph exceeds chunk size, split by sentences
                sentences = KOREAN_SENTENCE_SPLIT_REGEX.split(p_clean)
                current_accum = ""
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    if len(current_accum) + len(s_clean) + 1 <= self.chunk_size:
                        current_accum = f"{current_accum} {s_clean}".strip()
                    else:
                        if current_accum:
                            raw_sections.append(current_accum)
                        # If single sentence itself is larger than chunk_size, hard slice
                        if len(s_clean) > self.chunk_size:
                            for idx in range(0, len(s_clean), self.chunk_size - self.chunk_overlap):
                                raw_sections.append(s_clean[idx:idx + self.chunk_size])
                            current_accum = ""
                        else:
                            current_accum = s_clean
                if current_accum:
                    raw_sections.append(current_accum)

        # Merge raw sections into chunk_size windows with overlap
        chunks: List[str] = []
        current_chunk = ""

        for sec in raw_sections:
            if not current_chunk:
                current_chunk = sec
            elif len(current_chunk) + len(sec) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{sec}"
            else:
                chunks.append(current_chunk)
                # Keep overlap from the end of the previous chunk
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = f"{overlap_text}\n\n{sec}"

        if current_chunk and (not chunks or chunks[-1] != current_chunk):
            chunks.append(current_chunk)

        # Wrap into structured chunk dictionaries
        chunk_objects: List[Dict[str, Any]] = []
        total_chunks = len(chunks)

        for i, c_text in enumerate(chunks):
            chunk_meta = dict(metadata)
            chunk_meta.update({
                "chunk_index": i,
                "total_chunks": total_chunks,
                "char_length": len(c_text)
            })
            chunk_objects.append({
                "text": c_text,
                "metadata": chunk_meta
            })

        logger.info(f"Chunked document '{metadata.get('source', 'unknown')}' into {total_chunks} chunks.")
        return chunk_objects

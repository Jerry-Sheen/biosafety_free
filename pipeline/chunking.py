"""
Structure-Preserving Semantic Chunker for Biosafety Documents and Legal Texts.
Prioritizes:
1. Heading & Chapter/Section hierarchy
2. Legal articles (제1조, 제2조 등) and item enumerations (1., 가., (1))
3. Tables ([표 시작] ... [표 종료] protected from mid-table severance)
4. Natural paragraphs
5. Length constraints with semantic boundary overlap
"""
import re
import hashlib
from typing import List, Dict, Any, Optional
from config import config
from parsers import DocumentRecord
from pipeline.logger import logger

# Regex for Korean legal articles, chapters, and enumerations
LEGAL_ARTICLE_REGEX = re.compile(
    r"(?=(?:^|\n)(?:제\s*\d+\s*조(?:의\s*\d+)?\s*(?:\([^)]+\))?|제\s*\d+\s*장|제\s*\d+\s*절))"
)
ENUM_ITEM_REGEX = re.compile(
    r"(?=(?:^|\n)(?:\d+\.|\([0-9가-힣]+\)|[가-힣]\.|[①-⑳]))"
)
TABLE_BLOCK_REGEX = re.compile(r"(\[표 시작.*?\[표 종료\])", re.DOTALL)


def compute_chunk_hash(text: str) -> str:
    """Computes a normalized SHA-256 hash of chunk text for deduplication."""
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class StructurePreservingChunker:
    """Chunks documents while strictly defending tables, articles, and section integrity."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None
    ):
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.min_chunk_size = min_chunk_size or config.min_chunk_size

    def chunk_records(self, records: List[DocumentRecord]) -> List[Dict[str, Any]]:
        """
        Processes a list of DocumentRecords into structured chunk dictionaries.
        """
        all_chunks: List[Dict[str, Any]] = []

        for record in records:
            meta = record.metadata
            content_type = meta.get("content_type", "body")
            text = record.text.strip()
            if not text:
                continue

            # Case 1: Standalone Table records
            # Protect tables from splitting if reasonable size, or split by rows if oversized
            if content_type == "table":
                table_chunks = self._chunk_table_record(text, meta)
                all_chunks.extend(table_chunks)
                continue

            # Case 2: Body records (may contain embedded tables or legal articles)
            body_chunks = self._chunk_body_record(text, meta)
            all_chunks.extend(body_chunks)

        # Assign global chunk indices
        for idx, ch in enumerate(all_chunks):
            ch["metadata"]["chunk_index"] = idx
            ch["metadata"]["total_chunks"] = len(all_chunks)

        logger.info(f"Chunked {len(records)} records into {len(all_chunks)} structure-preserved chunks.")
        return all_chunks

    def _chunk_table_record(self, table_text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Keeps table together or splits along row boundaries if exceptionally large."""
        if len(table_text) <= self.chunk_size * 1.5:
            c_hash = compute_chunk_hash(table_text)
            c_meta = dict(meta)
            c_meta.update({
                "page_start": meta.get("page", 1),
                "page_end": meta.get("page", 1),
                "chunk_hash": c_hash,
                "content_type": "table"
            })
            return [{"text": table_text, "metadata": c_meta}]

        # Large table split by markdown row boundaries
        lines = table_text.splitlines()
        chunks = []
        header_lines = lines[:2] if len(lines) >= 2 and "---" in lines[1] else []
        row_lines = lines[2:] if header_lines else lines

        current_rows = []
        current_len = sum(len(h) for h in header_lines)

        for row in row_lines:
            if current_len + len(row) + 1 > self.chunk_size and current_rows:
                chunk_body = "\n".join(header_lines + current_rows)
                if not chunk_body.startswith("[표 시작]"):
                    chunk_body = f"[표 시작]\n{chunk_body}"
                if not chunk_body.endswith("[표 종료]"):
                    chunk_body = f"{chunk_body}\n[표 종료]"
                c_hash = compute_chunk_hash(chunk_body)
                c_meta = dict(meta)
                c_meta.update({
                    "page_start": meta.get("page", 1),
                    "page_end": meta.get("page", 1),
                    "chunk_hash": c_hash,
                    "content_type": "table"
                })
                chunks.append({"text": chunk_body, "metadata": c_meta})
                current_rows = [row]
                current_len = sum(len(h) for h in header_lines) + len(row)
            else:
                current_rows.append(row)
                current_len += len(row) + 1

        if current_rows:
            chunk_body = "\n".join(header_lines + current_rows)
            if not chunk_body.startswith("[표 시작]"):
                chunk_body = f"[표 시작]\n{chunk_body}"
            if not chunk_body.endswith("[표 종료]"):
                chunk_body = f"{chunk_body}\n[표 종료]"
            c_hash = compute_chunk_hash(chunk_body)
            c_meta = dict(meta)
            c_meta.update({
                "page_start": meta.get("page", 1),
                "page_end": meta.get("page", 1),
                "chunk_hash": c_hash,
                "content_type": "table"
            })
            chunks.append({"text": chunk_body, "metadata": c_meta})

        return chunks

    def _chunk_body_record(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunks body text along legal articles, paragraphs, and sentence boundaries."""
        # 1. First split by legal articles if present
        article_blocks = [b.strip() for b in LEGAL_ARTICLE_REGEX.split(text) if b.strip()]
        if not article_blocks:
            article_blocks = [text]

        atomic_units: List[str] = []

        for block in article_blocks:
            # 2. Check for embedded tables inside the block
            parts = TABLE_BLOCK_REGEX.split(block)
            for part in parts:
                part_clean = part.strip()
                if not part_clean:
                    continue
                # If table block, treat as single atomic unit
                if part_clean.startswith("[표 시작"):
                    atomic_units.append(part_clean)
                    continue

                # 3. Split by natural paragraphs
                paras = [p.strip() for p in part_clean.split("\n\n") if p.strip()]
                for para in paras:
                    if len(para) <= self.chunk_size:
                        atomic_units.append(para)
                    else:
                        # 4. Paragraph is large: split by enumeration items or sentences
                        items = [it.strip() for it in ENUM_ITEM_REGEX.split(para) if it.strip()]
                        if len(items) > 1:
                            for it in items:
                                if len(it) <= self.chunk_size:
                                    atomic_units.append(it)
                                else:
                                    atomic_units.extend(self._split_sentences(it))
                        else:
                            atomic_units.extend(self._split_sentences(para))

        # 5. Merge atomic units into chunk_size windows with overlap
        merged_chunks: List[str] = []
        current_chunk = ""

        for unit in atomic_units:
            if not current_chunk:
                current_chunk = unit
            elif len(current_chunk) + len(unit) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{unit}"
            else:
                merged_chunks.append(current_chunk)
                # Keep semantic overlap from the end
                if len(current_chunk) > self.chunk_overlap and not current_chunk.endswith("[표 종료]"):
                    overlap = current_chunk[-self.chunk_overlap:].strip()
                    current_chunk = f"{overlap}\n\n{unit}"
                else:
                    current_chunk = unit

        if current_chunk:
            merged_chunks.append(current_chunk)

        # Wrap in chunk objects
        page_num = meta.get("page", 1)
        result: List[Dict[str, Any]] = []

        for ch_text in merged_chunks:
            c_hash = compute_chunk_hash(ch_text)
            c_meta = dict(meta)
            c_meta.update({
                "page_start": page_num,
                "page_end": page_num,
                "chunk_hash": c_hash,
                "char_length": len(ch_text)
            })
            result.append({"text": ch_text, "metadata": c_meta})

        return result

    def _split_sentences(self, text: str) -> List[str]:
        """Splits text by sentence terminators while protecting decimals and acronyms."""
        sentence_regex = re.compile(r"(?<=[.!?다요죠])\s+")
        sentences = [s.strip() for s in sentence_regex.split(text) if s.strip()]
        result = []
        current = ""

        for s in sentences:
            if not current:
                current = s
            elif len(current) + len(s) + 1 <= self.chunk_size:
                current = f"{current} {s}"
            else:
                result.append(current)
                # If single sentence exceeds chunk size, slice safely
                if len(s) > self.chunk_size:
                    for i in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        result.append(s[i:i + self.chunk_size])
                    current = ""
                else:
                    current = s

        if current:
            result.append(current)

        return result

"""
Markdown Document Parser for Biosafety RAG Pipeline.
Preserves heading hierarchy (# H1, ## H2, ### H3), keeps code blocks, tables,
and bullet lists intact, and tracks section paths in metadata.
"""
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

from config import config
from parsers import DocumentRecord
from pipeline.logger import logger

HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownParser:
    """Parses markdown files while preserving headings, section paths, and structured blocks."""

    @classmethod
    def parse(cls, file_path: Path) -> List[DocumentRecord]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_bytes = file_path.read_bytes()
        doc_hash = hashlib.sha256(file_bytes).hexdigest()
        source_name = file_path.name
        source_path = str(file_path.resolve())

        # Read text with encoding fallback
        content = None
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                content = file_path.read_text(encoding=enc)
                break
            except Exception:
                continue
        if content is None:
            content = file_path.read_text(encoding="utf-8", errors="replace")

        # Split into sections based on headings
        lines = content.splitlines()
        sections: List[Tuple[str, List[str], str]] = []  # (heading_path, lines, content_type)

        current_hierarchy: List[str] = []
        current_lines: List[str] = []
        doc_title = source_name

        in_code_block = False

        for line in lines:
            stripped = line.strip()
            # Toggle code block state
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            heading_match = HEADING_REGEX.match(line) if not in_code_block else None
            if heading_match:
                # Flush previous section
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        section_path = " > ".join(current_hierarchy) if current_hierarchy else "Introduction"
                        sections.append((section_path, current_lines.copy(), "body"))
                    current_lines = []

                hashes, title = heading_match.groups()
                level = len(hashes)
                title_clean = title.strip()

                if level == 1 and not doc_title:
                    doc_title = title_clean

                # Update hierarchy
                if len(current_hierarchy) >= level:
                    current_hierarchy = current_hierarchy[:level - 1]
                current_hierarchy.append(title_clean)
                current_lines.append(line)
            else:
                current_lines.append(line)

        # Flush final section
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                section_path = " > ".join(current_hierarchy) if current_hierarchy else "General"
                sections.append((section_path, current_lines, "body"))

        records: List[DocumentRecord] = []
        for idx, (sec_path, sec_lines, c_type) in enumerate(sections):
            sec_text = "\n".join(sec_lines).strip()
            is_table_section = "[표 시작]" in sec_text or (sec_text.count("|") >= 4 and "---" in sec_text)
            actual_type = "table" if is_table_section else "body"

            meta = {
                "source": source_name,
                "source_path": source_path,
                "document_title": doc_title,
                "section": sec_path,
                "page": idx + 1,
                "content_type": actual_type,
                "extractor": "markdown_hierarchical",
                "document_hash": doc_hash
            }
            records.append(DocumentRecord(text=sec_text, raw_text=sec_text, metadata=meta))

        if not records and content.strip():
            # If no headings found, wrap the whole markdown
            meta = {
                "source": source_name,
                "source_path": source_path,
                "document_title": doc_title,
                "section": "Main",
                "page": 1,
                "content_type": "body",
                "extractor": "markdown_flat",
                "document_hash": doc_hash
            }
            records.append(DocumentRecord(text=content.strip(), raw_text=content.strip(), metadata=meta))

        return records

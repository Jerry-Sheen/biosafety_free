from pathlib import Path
from typing import Dict, Any, List, Optional
from src.security.logger import get_secure_logger
from .record import DocumentRecord

logger = get_secure_logger("docling_parser")

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    logger.warning("Docling package is not available. Please install 'docling' for advanced layout/OCR parsing.")


class DoclingParser:
    """Document parser leveraging IBM Docling with OCR capabilities.

    Returns a :class:`DocumentRecord` so that all parsers share a unified output.
    """

    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self._converter: Optional[Any] = None

    def _get_converter(self):
        """Lazily initialise DocumentConverter with appropriate OCR settings."""
        if not HAS_DOCLING:
            raise ImportError(
                "Docling is not installed in the environment. Install it via 'pip install docling'."
            )
        if self._converter is None:
            logger.info(f"Initializing Docling DocumentConverter (OCR Enabled: {self.enable_ocr})")
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.enable_ocr
            pipeline_options.do_table_structure = True
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            self._converter = DocumentConverter(format_options=format_options)
        return self._converter

    def parse(self, file_path: Path) -> DocumentRecord:
        """Parse a document with Docling and return a :class:`DocumentRecord`.

        The ``tables`` field is a list of dictionaries containing ``content`` (markdown)
        and ``metadata`` (including ``table_quality`` and ``original_image`` when applicable).
        """
        file_path = Path(file_path)
        logger.info(f"Running Docling parser on: {file_path.name}")
        converter = self._get_converter()
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        # Export full markdown text
        markdown_text = doc.export_to_markdown()

        # Extract tables with rich metadata
        tables: List[Dict[str, Any]] = []
        try:
            for idx, table in enumerate(getattr(doc, "tables", []), start=1):
                md = table.export_to_markdown()
                rows = [r for r in md.split("\n") if r.strip()]
                num_rows = max(len(rows) - 1, 0)  # subtract header separator if present
                num_cols = rows[0].count("|") - 1 if rows else 0
                meta = {
                    "content_type": "table",
                    "table_id": f"{file_path.stem}_table_{idx}",
                    "page": getattr(table, "page_number", None),
                    "rows": num_rows,
                    "columns": num_cols,
                }
                if num_rows == 0 or num_cols == 0:
                    meta["table_quality"] = "low"
                    # Record placeholder image path; actual image extraction would be implemented later.
                    meta["original_image"] = f"{file_path.stem}_table_{idx}.png"
                tables.append({"content": md, "metadata": meta})
        except Exception as e:
            logger.warning(f"Could not export tables directly: {e}")

        metadata = {
            "source": file_path.name,
            "parser": "docling",
            "file_size": file_path.stat().st_size,
            "ocr_applied": self.enable_ocr and file_path.suffix.lower() == ".pdf",
        }
        if hasattr(doc, "pages") and doc.pages:
            metadata["page_count"] = len(doc.pages)

        return DocumentRecord(text=markdown_text, metadata={**metadata, "tables": tables})

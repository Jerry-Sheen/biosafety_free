"""
Unified Master Document Parser.
Coordinates Docling (with OCR), HWP Parser, and Fallback Parsers.
Guarantees resilient processing across PDF, Excel, Word, CSV, HWP, and HWPX.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from src.parsers.docling_parser import DoclingParser, HAS_DOCLING
from src.parsers.hwp_parser import HWPParser
from src.parsers.fallback_parser import FallbackParser
from src.security.logger import get_secure_logger

logger = get_secure_logger("document_parser")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv",
    ".hwp", ".hwpx", ".txt", ".md", ".json"
}


class DocumentParser:
    """Master document parser with intelligent parser selection and fault tolerance."""

    def __init__(self, prefer_docling: bool = True, enable_ocr: bool = True):
        # Preserve flags; Docling may be unavailable.
        self.prefer_docling = prefer_docling and HAS_DOCLING
        self.enable_ocr = enable_ocr
        # Docling parser will be instantiated lazily per file because OCR needs may differ.
        self.docling_parser = None
        self.prefer_docling = prefer_docling and HAS_DOCLING
        self.enable_ocr = enable_ocr
        if self.prefer_docling:
            try:
                self.docling_parser = DoclingParser(enable_ocr=self.enable_ocr)
            except Exception as e:
                logger.warning(f"Failed to initialize Docling: {e}. Defaulting to fallback parser.")
                self.docling_parser = None
                self.prefer_docling = False
        else:
            self.docling_parser = None

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parses an uploaded file using the most appropriate parser.
        Returns:
            {
                "success": bool,
                "text": str,
                "metadata": dict,
                "tables": list,
                "error": Optional[str]
            }
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "success": False,
                "text": "",
                "metadata": {"source": file_path.name},
                "tables": [],
                "error": f"Unsupported file extension: {ext}. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            }

        logger.info(f"Dispatching parser for {file_path.name} (ext: {ext})")

        # 1. HWP / HWPX handling
        if ext in (".hwp", ".hwpx"):
            try:
                # First try native HWP parser (binary OLE2)
                result = HWPParser.parse(file_path)
                result["success"] = True
                result["error"] = None
                return result
            except Exception as e:
                logger.warning(f"HWP Parser failed on {file_path.name}: {e}")
                # Fallback: try HWPX parsing (zip based)
                try:
                    logger.info(f"Attempting HWPX parsing fallback for {file_path.name}")
                    result = HWPParser._parse_hwpx(file_path)
                    result["success"] = True
                    result["error"] = None
                    return result
                except Exception as e2:
                    logger.error(f"Both HWP and HWPX parsers failed on {file_path.name}: {e2}")
                    # Final fallback: raw binary → UTF‑8 ignore
                    try:
                        raw_bytes = file_path.read_bytes()
                        text = raw_bytes.decode("utf-8", errors="ignore")
                        logger.info(f"Raw fallback extraction succeeded for {file_path.name} (length {len(text)})")
                        return {
                            "text": text,
                            "metadata": {"source": file_path.name, "format": ext},
                            "tables": [],
                            "success": True,
                            "error": None,
                        }
                    except Exception as e3:
                        logger.error(f"Raw fallback also failed on {file_path.name}: {e3}")
                        return {
                            "success": False,
                            "text": "",
                            "metadata": {"source": file_path.name, "format": ext},
                            "tables": [],
                            "error": f"HWP parsing error: {str(e)}; fallback HWPX error: {str(e2)}; raw read error: {str(e3)}",
                        }


        # 4. JSON file handling
        if ext == ".json":
            try:
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 병원균 사전 등 키-값 딕셔너리인 경우 엔티티별 명확한 자연어/구조화 프로필로 변환
                if isinstance(data, dict):
                    entries = []
                    for key, val in data.items():
                        lines = [f"[병원균/항목: {key}]"]
                        if isinstance(val, dict):
                            for sub_k, sub_v in val.items():
                                lines.append(f"- {sub_k}: {sub_v}")
                        elif isinstance(val, list):
                            lines.append(f"- 목록: {', '.join(str(x) for x in val)}")
                        else:
                            lines.append(f"- 내용: {val}")
                        entries.append("\n".join(lines))
                    text = "\n\n".join(entries)
                elif isinstance(data, list):
                    entries = []
                    for item in data:
                        if isinstance(item, dict):
                            lines = []
                            # 대표 이름 찾기
                            name_val = item.get("name") or item.get("병원균") or item.get("title") or "항목"
                            lines.append(f"[{name_val}]")
                            for sub_k, sub_v in item.items():
                                lines.append(f"- {sub_k}: {sub_v}")
                            entries.append("\n".join(lines))
                        else:
                            entries.append(str(item))
                    text = "\n\n".join(entries)
                else:
                    text = json.dumps(data, ensure_ascii=False, indent=2)

                return {
                    "text": text,
                    "metadata": {"source": file_path.name, "format": ext},
                    "tables": [],
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                logger.error(f"JSON parsing failed on {file_path.name}: {e}")
                return {
                    "success": False,
                    "text": "",
                    "metadata": {"source": file_path.name, "format": ext},
                    "tables": [],
                    "error": f"JSON parsing error: {str(e)}",
                }

        if self.prefer_docling and self.docling_parser and ext in (".pdf", ".docx", ".xlsx"):
            try:
                logger.info(f"Attempting Docling parsing for {file_path.name}")
                result = self.docling_parser.parse(file_path)
                result["success"] = True
                result["error"] = None
                return result
            except Exception as e:
                logger.warning(f"Docling parsing failed on {file_path.name}: {e}. Retrying with fallback parser.")

        # 3. Fallback Parser (PDF, DOCX, XLSX, CSV, TXT, MD)
        try:
            result = FallbackParser.parse(file_path)
            result["success"] = True
            result["error"] = None
            return result
        except Exception as e:
            logger.error(f"Fallback parsing failed on {file_path.name}: {e}")
            return {
                "success": False,
                "text": "",
                "metadata": {"source": file_path.name, "format": ext},
                "tables": [],
                "error": f"Parsing failed: {str(e)}"
            }

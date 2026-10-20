"""
HWP and HWPX Parser for Korean Hangul Documents.
Extracts clean text, preserves paragraphs and table structures, and returns DocumentRecord items.
"""
import zipfile
import zlib
import struct
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import config
from parsers import DocumentRecord
from pipeline.logger import logger, IngestionErrorType, log_pipeline_event

try:
    import olefile
    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False
    logger.warning("olefile is not installed. Standard .hwp binary parsing will be limited.")


class HWPParser:
    """Specialized parser for HWP binary and HWPX zip+xml formats."""

    @classmethod
    def parse(cls, file_path: Path) -> List[DocumentRecord]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_bytes = file_path.read_bytes()
        doc_hash = hashlib.sha256(file_bytes).hexdigest()
        ext = file_path.suffix.lower()

        if ext == ".hwpx":
            return cls._parse_hwpx(file_path, doc_hash)
        elif ext == ".hwp":
            return cls._parse_hwp(file_path, doc_hash)
        else:
            raise ValueError(f"Unsupported HWP extension: {ext}")

    @classmethod
    def _parse_hwpx(cls, file_path: Path, doc_hash: str) -> List[DocumentRecord]:
        logger.info(f"Parsing HWPX document: {file_path.name}")
        records: List[DocumentRecord] = []
        source_name = file_path.name
        source_path = str(file_path.resolve())

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                section_files = [
                    name for name in zf.namelist()
                    if name.startswith("Contents/section") and name.endswith(".xml")
                ]
                section_files.sort()

                for sec_idx, sec_name in enumerate(section_files):
                    xml_data = zf.read(sec_name)
                    root = ET.fromstring(xml_data)

                    # Extract paragraphs and tables
                    sec_texts = []
                    for elem in root.iter():
                        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                        if tag_name == "t" and elem.text:
                            sec_texts.append(elem.text.strip())

                    sec_body = "\n".join([t for t in sec_texts if t])
                    if sec_body:
                        records.append(DocumentRecord(
                            text=sec_body,
                            raw_text=sec_body,
                            metadata={
                                "source": source_name,
                                "source_path": source_path,
                                "page": sec_idx + 1,
                                "section": f"Section_{sec_idx}",
                                "content_type": "body",
                                "extractor": "hwpx_xml",
                                "document_hash": doc_hash,
                                "format": "hwpx"
                            }
                        ))
            return records
        except Exception as e:
            log_pipeline_event(
                filename=source_name,
                parser="hwpx_parser",
                error_type=IngestionErrorType.FILE_READ_ERROR,
                message=f"HWPX extraction failed: {e}",
                status="failed"
            )
            raise

    @classmethod
    def _parse_hwp(cls, file_path: Path, doc_hash: str) -> List[DocumentRecord]:
        logger.info(f"Parsing HWP binary document using HWP 5.0 record parser: {file_path.name}")
        if not HAS_OLEFILE:
            raise RuntimeError("olefile library is required to parse .hwp files. Please install olefile.")

        source_name = file_path.name
        source_path = str(file_path.resolve())
        records: List[DocumentRecord] = []

        try:
            ole = olefile.OleFileIO(str(file_path))
            dirs = ole.listdir()

            section_streams = [
                d for d in dirs if len(d) >= 2 and d[0] == "BodyText" and d[1].startswith("Section")
            ]
            section_streams.sort(
                key=lambda x: int(x[1].replace("Section", "")) if x[1].replace("Section", "").isdigit() else x[1]
            )

            for sec_idx, stream_path in enumerate(section_streams):
                data = ole.openstream(stream_path).read()

                # HWP section streams are raw deflate (wbits=-15)
                try:
                    decompressed = zlib.decompress(data, -15)
                except Exception:
                    try:
                        decompressed = zlib.decompress(data)
                    except Exception:
                        decompressed = data

                offset = 0
                buf_len = len(decompressed)
                extracted_paragraphs: List[str] = []

                while offset + 4 <= buf_len:
                    header = struct.unpack("<I", decompressed[offset:offset + 4])[0]
                    tag_id = header & 0x3FF
                    size = (header >> 20) & 0xFFF
                    offset += 4

                    if size == 0xFFF:
                        if offset + 4 > buf_len:
                            break
                        size = struct.unpack("<I", decompressed[offset:offset + 4])[0]
                        offset += 4

                    record_data = decompressed[offset:offset + size]
                    offset += size

                    # Tag 67: HWPTAG_PARA_TEXT (Paragraph text stream)
                    if tag_id == 67:
                        chars: List[str] = []
                        idx = 0
                        rec_len = len(record_data)

                        while idx + 2 <= rec_len:
                            code = struct.unpack("<H", record_data[idx:idx + 2])[0]
                            idx += 2

                            # Inline extended control bytes (16 bytes)
                            if code in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
                                idx += 16
                            elif code in (4, 5, 6, 7, 8, 19, 20):
                                pass
                            elif code == 9:
                                chars.append("\t")
                            elif code == 10:
                                chars.append("\n")
                            elif code == 24:
                                chars.append("-")
                            elif code in (30, 31):
                                chars.append(" ")
                            elif 32 <= code < 0xD800 or 0xE000 <= code <= 0x10FFFF:
                                chars.append(chr(code))

                        p_text = "".join(chars).strip()
                        if p_text:
                            clean_p = p_text.encode("utf-8", "ignore").decode("utf-8").strip()
                            if clean_p:
                                extracted_paragraphs.append(clean_p)

                if extracted_paragraphs:
                    sec_text = "\n\n".join(extracted_paragraphs)
                    records.append(DocumentRecord(
                        text=sec_text,
                        raw_text=sec_text,
                        metadata={
                            "source": source_name,
                            "source_path": source_path,
                            "page": sec_idx + 1,
                            "section": f"Section_{sec_idx}",
                            "content_type": "body",
                            "extractor": "hwp_ole",
                            "document_hash": doc_hash,
                            "format": "hwp",
                            "paragraphs_count": len(extracted_paragraphs)
                        }
                    ))

            ole.close()
            return records
        except Exception as e:
            log_pipeline_event(
                filename=source_name,
                parser="hwp_parser",
                error_type=IngestionErrorType.FILE_READ_ERROR,
                message=f"HWP extraction failed: {e}",
                status="failed"
            )
            raise

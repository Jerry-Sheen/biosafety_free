"""
HWP and HWPX document parser for Korean office documents.
Supports:
- .hwpx: XML-based Hangul documents (unzipped XML stream extraction)
- .hwp: OLE binary Hangul documents (OLE stream decompression & UTF-16LE decoding)
"""
import zipfile
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List
from src.security.logger import get_secure_logger

logger = get_secure_logger("hwp_parser")

try:
    import olefile
    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False
    logger.warning("olefile is not installed. Standard .hwp binary parsing will be limited.")


class HWPParser:
    """Extracts text, metadata, and tables from .hwp and .hwpx files."""

    @classmethod
    def parse(cls, file_path: Path) -> Dict[str, Any]:
        """
        Parses an HWP or HWPX file and returns structured output.
        Returns: {
            "text": str,
            "metadata": dict,
            "tables": list,
            "format": "hwpx" or "hwp"
        }
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".hwpx":
            return cls._parse_hwpx(file_path)
        elif ext == ".hwp":
            return cls._parse_hwp(file_path)
        else:
            raise ValueError(f"Unsupported HWP format: {ext}")

    @classmethod
    def _parse_hwpx(cls, file_path: Path) -> Dict[str, Any]:
        """Extracts text from HWPX (Zip + XML)."""
        logger.info(f"Parsing HWPX document: {file_path.name}")
        full_text_parts: List[str] = []
        tables: List[str] = []

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                section_files = [name for name in zf.namelist() if name.startswith("Contents/section") and name.endswith(".xml")]
                section_files.sort()

                for sec_name in section_files:
                    xml_data = zf.read(sec_name)
                    root = ET.fromstring(xml_data)

                    # Extract text elements (typically under tags ending in 't')
                    sec_texts = []
                    for elem in root.iter():
                        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                        if tag_name == "t" and elem.text:
                            sec_texts.append(elem.text)

                    if sec_texts:
                        full_text_parts.append("\n".join(sec_texts))

            text = "\n\n".join(full_text_parts)
            return {
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "format": "hwpx",
                    "file_size": file_path.stat().st_size
                },
                "tables": tables
            }
        except Exception as e:
            logger.error(f"Error parsing HWPX {file_path.name}: {e}")
            raise

    @classmethod
    def _parse_hwp(cls, file_path: Path) -> Dict[str, Any]:
        """
        Extracts clean text from binary HWP (OLE storage) using HWP 5.0 record specifications.
        Selectively parses HWPTAG_PARA_TEXT (Tag 67) and skips inline extended control bytes
        to prevent binary metadata from being misdecoded into corrupt Korean characters.
        """
        import struct
        logger.info(f"Parsing HWP binary document using HWP 5.0 record parser: {file_path.name}")
        if not HAS_OLEFILE:
            raise RuntimeError("olefile library is required to parse .hwp files. Please install olefile.")

        extracted_paragraphs: List[str] = []

        try:
            ole = olefile.OleFileIO(str(file_path))
            dirs = ole.listdir()

            # Find all BodyText/Section streams in correct order
            section_streams = [
                d for d in dirs if len(d) >= 2 and d[0] == "BodyText" and d[1].startswith("Section")
            ]
            # Natural sort section streams (e.g. Section0, Section1, ...)
            section_streams.sort(key=lambda x: int(x[1].replace("Section", "")) if x[1].replace("Section", "").isdigit() else x[1])

            for stream_path in section_streams:
                stream_name = "/".join(stream_path)
                data = ole.openstream(stream_path).read()

                # HWP section streams are compressed with Deflate without zlib wrapper (raw deflate, wbits=-15)
                try:
                    decompressed = zlib.decompress(data, -15)
                except Exception:
                    try:
                        decompressed = zlib.decompress(data)
                    except Exception:
                        decompressed = data

                # Parse HWP 5.0 Records
                offset = 0
                buf_len = len(decompressed)
                while offset + 4 <= buf_len:
                    header = struct.unpack("<I", decompressed[offset:offset + 4])[0]
                    tag_id = header & 0x3FF
                    level = (header >> 10) & 0x3FF
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

                            # Extended and object control characters contain 16 bytes (8 words) of inline binary metadata
                            # 1, 2, 3: Field start/end, bookmarks
                            # 11~18: Drawings, tables, pictures, equations, footnotes, headers/footers
                            # 21~23: Object controls
                            if code in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
                                idx += 16
                            elif code in (4, 5, 6, 7, 8, 19, 20):
                                # Other control codes (skip without additional data)
                                pass
                            elif code == 9:
                                chars.append("\t")
                            elif code == 10:
                                chars.append("\n")
                            elif code == 13:
                                # Paragraph break handled per record
                                pass
                            elif code == 24:
                                chars.append("-")
                            elif code in (30, 31):
                                chars.append(" ")
                            elif 32 <= code < 0xD800 or 0xE000 <= code <= 0x10FFFF:
                                chars.append(chr(code))

                        p_text = "".join(chars).strip()
                        if p_text:
                            # Sanitize any unexpected non-encodable characters
                            clean_p = p_text.encode("utf-8", "ignore").decode("utf-8").strip()
                            if clean_p:
                                extracted_paragraphs.append(clean_p)

            ole.close()
            combined_text = "\n\n".join(extracted_paragraphs)
            combined_text = combined_text.encode("utf-8", "ignore").decode("utf-8")
            logger.info(f"Successfully extracted {len(extracted_paragraphs)} paragraphs from {file_path.name}")
            return {
                "text": combined_text,
                "metadata": {
                    "source": file_path.name,
                    "format": "hwp",
                    "sections_count": len(section_streams),
                    "paragraphs_count": len(extracted_paragraphs),
                    "file_size": file_path.stat().st_size
                },
                "tables": []
            }
        except Exception as e:
            logger.error(f"Error parsing HWP binary file {file_path.name}: {e}")
            raise

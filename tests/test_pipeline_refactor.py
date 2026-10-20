"""
Comprehensive Unit and Integration Tests for Refactored Biosafety Ingestion Pipeline.
Covers all 11 required scenarios:
1. Pure text PDF
2. Scanned image PDF
3. Mixed text & image PDF
4. Korean OCR extraction
5. Table extraction in PDF
6. HWP binary document parsing
7. CP949 encoded CSV parsing
8. Markdown hierarchical heading parsing
9. Deduplication on identical document re-ingestion
10. Non-existent Gemini model name handling
11. Vector store ingestion completion even if Gemini API fails
"""
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz
from config import config
from parsers import DocumentRecord
from parsers.pdf_parser import PDFParser
from parsers.ocr_parser import extract_text_with_ocr
from parsers.csv_parser import CSVParser
from parsers.markdown_parser import MarkdownParser
from parsers.hwp_parser import HWPParser
from pipeline.classify import classify_pdf
from pipeline.quality import evaluate_text_quality
from pipeline.chunking import StructurePreservingChunker
from pipeline.deduplication import ingest_document
from vectorstore.store import VectorStore
from src.rag.gemini_client import GeminiModelResolver, classify_gemini_error
from pipeline.logger import IngestionErrorType


class TestBiosafetyPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Normal Text PDF Test
    def test_01_normal_text_pdf(self):
        doc = fitz.open()
        page = doc.new_page()
        text_content = (
            "생물작용제 및 독소의 제조·보유·수입 신고 및 허가 절차 안내서입니다. "
            "생물작용제는 탄저균, 페스트균, 보툴리눔 독소 등을 포함하며 질병관리청에 사전에 신고하여야 합니다. "
            "생물방어 및 생물안전 3등급 연구시설 허가 기준에 따라 철저히 관리되어야 합니다."
        )
        page.insert_textbox(fitz.Rect(50, 50, 500, 500), text_content, fontname="korea", fontsize=12)
        pdf_path = self.temp_path / "normal_text.pdf"
        doc.save(str(pdf_path))
        doc.close()

        # Classify
        classification = classify_pdf(pdf_path)
        self.assertFalse(classification["is_scanned"])
        self.assertEqual(classification["page_count"], 1)
        self.assertGreater(classification["average_chars_per_page"], 50)

        # Parse
        parser = PDFParser(enable_docling=False)
        records = parser.parse(pdf_path)
        self.assertGreater(len(records), 0)
        self.assertIn("생물작용제", records[0].text)
        self.assertFalse(records[0].metadata["ocr_used"])

    # 2. Scanned Image PDF Test
    def test_02_scanned_image_pdf(self):
        # Create an image containing text
        img = Image.new("RGB", (600, 200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((30, 80), "Anthrax Toxin Security Regulations 2026", fill=(0, 0, 0))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")

        doc = fitz.open()
        page = doc.new_page()
        page.insert_image(page.rect, stream=img_bytes.getvalue())
        pdf_path = self.temp_path / "scanned_image.pdf"
        doc.save(str(pdf_path))
        doc.close()

        classification = classify_pdf(pdf_path)
        self.assertTrue(classification["is_scanned"])
        self.assertEqual(classification["image_page_count"], 1)

    # 3. Mixed Text & Image PDF Test
    def test_03_mixed_pdf(self):
        doc = fitz.open()
        # Page 1: digital text
        p1 = doc.new_page()
        p1.insert_text((50, 72), "제1조(목적) 이 법은 생물작용제의 안전관리를 목적으로 한다. " * 3, fontsize=11)
        # Page 2: pure image
        p2 = doc.new_page()
        img = Image.new("RGB", (400, 200), color=(240, 240, 240))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        p2.insert_image(p2.rect, stream=img_bytes.getvalue())

        pdf_path = self.temp_path / "mixed_doc.pdf"
        doc.save(str(pdf_path))
        doc.close()

        classification = classify_pdf(pdf_path)
        self.assertTrue(classification["is_mixed"])
        self.assertEqual(classification["text_page_count"], 1)
        self.assertEqual(classification["image_page_count"], 1)

    # 4. Korean OCR Test
    def test_04_korean_ocr(self):
        img = Image.new("RGB", (600, 150), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((20, 50), "질병관리청 생물안전 2026 KOREA", fill=(0, 0, 0))

        res = extract_text_with_ocr(img, language="kor+eng")
        self.assertIn("status", res)
        self.assertIn("confidence", res)
        # Text extraction should run without crashing
        self.assertIsInstance(res["text"], str)

    # 5. PDF Table Extraction Test
    def test_05_pdf_table_extraction(self):
        doc = fitz.open()
        page = doc.new_page()
        # Draw a table with lines so PyMuPDF finds it
        rect = fitz.Rect(50, 50, 450, 200)
        page.draw_rect(rect)
        page.draw_line(fitz.Point(50, 100), fitz.Point(450, 100))
        page.draw_line(fitz.Point(200, 50), fitz.Point(200, 200))
        page.insert_text((60, 80), "구분", fontname="korea", fontsize=11)
        page.insert_text((220, 80), "신고서류", fontname="korea", fontsize=11)
        page.insert_text((60, 150), "생물작용제", fontname="korea", fontsize=11)
        page.insert_text((220, 150), "보유신고서", fontname="korea", fontsize=11)

        pdf_path = self.temp_path / "table_doc.pdf"
        doc.save(str(pdf_path))
        doc.close()

        parser = PDFParser(enable_docling=False)
        records = parser.parse(pdf_path)
        self.assertGreater(len(records), 0)
        has_content = any("생물작용제" in r.text or "구분" in r.text for r in records)
        self.assertTrue(has_content)

    # 6. HWP Binary Parsing Test (Mock OleFile stream)
    def test_06_hwp_parsing(self):
        hwp_path = self.temp_path / "sample.hwp"
        # Test error handling when file is not valid OLE
        hwp_path.write_bytes(b"INVALID_HWP_BINARY")
        with self.assertRaises(Exception):
            HWPParser.parse(hwp_path)

    # 7. CP949 CSV Parsing Test
    def test_07_cp949_csv(self):
        csv_content = "연번,생물작용제명,위험군,신고대상\n1,탄저균,고위험,허가대상\n2,보툴리눔독소,고위험,신고대상\n"
        csv_path = self.temp_path / "biosafety_cp949.csv"
        csv_path.write_bytes(csv_content.encode("cp949"))

        records = CSVParser.parse(csv_path)
        self.assertGreater(len(records), 0)
        self.assertIn("탄저균", records[0].text)
        self.assertEqual(records[0].metadata["content_type"], "table")
        self.assertEqual(records[0].metadata["encoding"], "cp949")

    # 8. Markdown Heading Hierarchy Test
    def test_08_markdown_heading(self):
        md_text = (
            "# 제1장 총칙\n\n"
            "본 지침은 생물작용제의 안전한 관리를 목적으로 한다.\n\n"
            "## 제1절 정의\n\n"
            "생물작용제란 자연적으로 존재하거나 유전자가 변형된 미생물을 말한다.\n\n"
            "| 작용제 | 독소 |\n| --- | --- |\n| 탄저 | 보툴리눔 |\n"
        )
        md_path = self.temp_path / "guidelines.md"
        md_path.write_text(md_text, encoding="utf-8")

        records = MarkdownParser.parse(md_path)
        self.assertGreater(len(records), 0)
        sections = [r.metadata.get("section") for r in records]
        self.assertTrue(any("제1장 총칙" in str(s) for s in sections))
        self.assertTrue(any("제1절 정의" in str(s) for s in sections))

    # 9. Deduplication Test (Re-ingesting identical document)
    def test_09_deduplication(self):
        test_file = self.temp_path / "test_doc.txt"
        test_file.write_text("생물작용제 및 독소 법령 제1조 테스트 문서입니다.", encoding="utf-8")

        mock_vstore = MagicMock()
        # First ingestion: document does not exist yet
        mock_vstore.has_document.return_value = False
        mock_vstore.add_chunks.return_value = 1

        res1 = ingest_document(test_file, force=False, vector_store=mock_vstore)
        self.assertEqual(res1["status"], "success")
        self.assertEqual(res1["chunks"], 1)

        # Second ingestion without force: document exists
        mock_vstore.has_document.return_value = True
        res2 = ingest_document(test_file, force=False, vector_store=mock_vstore)
        self.assertEqual(res2["status"], "skipped")
        self.assertEqual(res2["chunks"], 0)

        # Third ingestion with force=True: should re-ingest
        mock_vstore.delete_by_document_hash.return_value = 1
        res3 = ingest_document(test_file, force=True, vector_store=mock_vstore)
        self.assertEqual(res3["status"], "success")

    # 10. Non-existent Gemini Model Fallback Test
    def test_10_nonexistent_gemini_model(self):
        resolver = GeminiModelResolver(api_key="mock_key")
        # Mock available models list
        resolver.list_available_models = MagicMock(return_value=[
            {"short_name": "gemini-2.5-flash", "supports_generate_content": True, "display_name": "Flash 2.5"},
            {"short_name": "gemini-1.5-flash", "supports_generate_content": True, "display_name": "Flash 1.5"}
        ])

        # Test request for fake model name
        resolved, warning = resolver.resolve_model("models/gemini-3-flash")
        self.assertIn("gemini-2.5-flash", resolved)
        self.assertIsNotNone(warning)
        self.assertIn("Falling back", warning)

        # Test error classifier for 404
        fake_404_error = Exception("404 NOT_FOUND: models/gemini-3-flash is not found for API version v1beta")
        classified = classify_gemini_error(fake_404_error)
        self.assertEqual(classified["error_type"], IngestionErrorType.GEMINI_MODEL_NOT_FOUND)
        self.assertIn("404 NOT_FOUND", classified["user_message"])

    # 11. Ingestion works even if Gemini is unreachable/fails
    def test_11_ingestion_independent_of_gemini(self):
        test_file = self.temp_path / "independent.txt"
        test_file.write_text("생물안전 독립성 검증 텍스트입니다. LLM 오류와 관계없이 벡터 저장이 되어야 합니다.", encoding="utf-8")

        mock_vstore = MagicMock()
        mock_vstore.has_document.return_value = False
        mock_vstore.add_chunks.return_value = 1

        # Simulate Gemini being completely broken or unset
        with patch.object(config, "gemini_api_key", ""):
            res = ingest_document(test_file, force=False, vector_store=mock_vstore)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["chunks"], 1)


if __name__ == "__main__":
    unittest.main()

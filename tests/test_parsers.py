"""
Unit tests for parsers.
"""
from pathlib import Path
import tempfile
from src.parsers.fallback_parser import FallbackParser
from src.parsers.document_parser import DocumentParser


def test_fallback_parser_text():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("안녕하세요, 이것은 RAG 테스트 문서입니다.\n두 번째 줄 내용입니다.")
        f_path = Path(f.name)

    try:
        res = FallbackParser.parse(f_path)
        assert "안녕하세요" in res["text"]
        assert res["metadata"]["format"] == "txt"
    finally:
        if f_path.exists():
            f_path.unlink()


def test_fallback_parser_csv():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("이름,부서,직급\n홍길동,AI연구소,선임연구원\n김철수,보안팀,팀장")
        f_path = Path(f.name)

    try:
        res = FallbackParser.parse(f_path)
        assert "홍길동" in res["text"]
        assert "선임연구원" in res["text"]
        assert res["metadata"]["row_count"] == 2
    finally:
        if f_path.exists():
            f_path.unlink()


def test_document_parser_unsupported():
    parser = DocumentParser(prefer_docling=False)
    fake_path = Path("sample.invalid_ext")
    res = parser.parse_file(fake_path)
    assert res["success"] is False
    assert "Unsupported file extension" in res["error"]

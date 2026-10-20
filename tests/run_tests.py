"""
Standard library unittest runner to test core modules without external test runners.
"""
import sys
import unittest
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.security.secrets_manager import SecretsManager
from src.security.logger import SensitiveDataFilter
from src.rag.chunker import DocumentChunker
from src.parsers.fallback_parser import FallbackParser
from src.parsers.document_parser import DocumentParser


class TestRagVibeCore(unittest.TestCase):

    def test_mask_secret(self):
        sm = SecretsManager()
        self.assertEqual(sm.mask_secret("AIzaSyDummySecretKey1234567890"), "AIzaSy...7890")
        self.assertEqual(sm.mask_secret("short"), "********")
        self.assertEqual(sm.mask_secret(None), "NOT_CONFIGURED")

    def test_sensitive_data_filter(self):
        filt = SensitiveDataFilter()
        raw = "Connecting with key AIzaSyDUMMYKEY12345678901234567890123"
        sanitized = filt.sanitize(raw)
        self.assertIn("[REDACTED_SECRET]", sanitized)
        self.assertNotIn("AIzaSy", sanitized)

    def test_chunker_korean_text(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        sample = (
            "인공지능 기반 검색 증강 생성(RAG) 시스템은 최신 문서를 참조하여 환각 현상을 억제합니다. "
            "한국어 자연어 처리는 형태소와 조사의 결합 특성상 어절 단위의 분할이 매우 중요합니다. "
            "벡터 데이터베이스는 텍스트를 고차원 임베딩 벡터로 변환하여 밀집 검색을 수행합니다."
        )
        chunks = chunker.chunk_document(sample, {"source": "test.pdf"})
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["metadata"]["source"], "test.pdf")

    def test_fallback_parser_text(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("안녕하세요, 이것은 RAG 테스트 문서입니다.")
            f_path = Path(f.name)
        try:
            res = FallbackParser.parse(f_path)
            self.assertIn("안녕하세요", res["text"])
        finally:
            if f_path.exists():
                f_path.unlink()

    def test_document_parser_unsupported(self):
        parser = DocumentParser(prefer_docling=False)
        res = parser.parse_file(Path("test.invalid"))
        self.assertFalse(res["success"])
        self.assertIn("Unsupported", res["error"])

    def test_vector_store_if_chroma_installed(self):
        try:
            from src.rag.vector_store import PersistentVectorStore
            from src.rag.embeddings import LocalEmbeddingProvider

            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
                embedder = LocalEmbeddingProvider()
                vs = PersistentVectorStore(
                    persist_directory=Path(tmp_dir),
                    collection_name="test_col",
                    embedding_provider=embedder
                )
                chunks = [{"text": "테스트 문장입니다.", "metadata": {"source": "sample.pdf", "chunk_index": 0}}]
                added = vs.add_documents(chunks)
                self.assertEqual(added, 1)
                self.assertEqual(vs.get_stats()["total_chunks"], 1)
        except ImportError:
            print("Skipping vector store test (chromadb not installed in host environment).")


if __name__ == "__main__":
    unittest.main()

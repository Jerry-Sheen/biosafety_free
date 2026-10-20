"""
Unit tests for Document Chunker.
"""
from src.rag.chunker import DocumentChunker


def test_chunker_korean_text():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    korean_sample = (
        "인공지능 기반 검색 증강 생성(RAG) 시스템은 최신 문서를 참조하여 환각 현상을 억제합니다. "
        "한국어 자연어 처리는 형태소와 조사의 결합 특성상 어절 단위의 분할이 매우 중요합니다. "
        "벡터 데이터베이스는 텍스트를 고차원 임베딩 벡터로 변환하여 밀집 검색을 수행합니다. "
        "이를 통해 사용자는 원하는 정보를 실시간으로 대화형 인터페이스를 통해 질의할 수 있습니다."
    )
    metadata = {"source": "test_doc.pdf", "category": "AI"}

    chunks = chunker.chunk_document(korean_sample, metadata)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["source"] == "test_doc.pdf"
        assert "chunk_index" in chunk["metadata"]
        assert "total_chunks" in chunk["metadata"]
        assert chunk["metadata"]["total_chunks"] == len(chunks)


def test_chunker_empty_text():
    chunker = DocumentChunker()
    chunks = chunker.chunk_document("", {"source": "empty.txt"})
    assert chunks == []

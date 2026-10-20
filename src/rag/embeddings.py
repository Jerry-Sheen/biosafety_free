'''
Local Embedding Provider Module (SentenceTransformers).

Ollama support has been completely removed.
admin.py (문서 색인) 와 user.py (질의 검색) 는 반드시 동일한 임베딩 모델을 사용해야
벡터 차원이 일치하고 기존 데이터가 보존됩니다. 따라서 임베딩 모델은
config.local_embed_model 하나로 고정되며, 프로세스 내에서 한 번만 로드(캐시)됩니다.
'''

import threading
from typing import List, Optional, Any

from src.config import config
from src.security.logger import get_secure_logger

logger = get_secure_logger("embeddings")

_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.Lock()


def _load_sentence_transformer(model_name: str):
    """Loads (once per process) and caches a SentenceTransformer model."""
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {model_name}")
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        return _MODEL_CACHE[model_name]


class LocalEmbeddingProvider:
    """Free, fully local embedding provider based on SentenceTransformers."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.local_embed_model
        self.model = _load_sentence_transformer(self.model_name)
        self._dimension = int(self.model.get_sentence_embedding_dimension())

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, documents: List[str], batch_size: int = 32) -> List[List[float]]:
        if not documents:
            return []
        vectors = self.model.encode(
            documents, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()


def get_embedding_provider() -> Any:
    """Returns the single fixed embedding provider shared by admin.py and user.py."""
    return LocalEmbeddingProvider()

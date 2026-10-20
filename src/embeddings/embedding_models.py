# src/embeddings/embedding_models.py
"""Utility for selecting and loading free embedding models.

The project originally used an Ollama model (`bge-m3`) for generating embeddings.
For users who prefer a faster, zero‑cost alternative we provide a small set of
open‑source models that can be loaded directly with the ``sentence‑transformers``
library.  All models listed below can be downloaded from Hugging Face and run on
CPU (or GPU if available) without any external service costs.

Available models
----------------
| Identifier   | Hugging‑Face model                     | Size / Speed |
|--------------|---------------------------------------|--------------|
| ``bge-m3``   | ``BAAI/bge-m3``                        | high quality |
| ``mini-l6``  | ``sentence‑transformers/all‑MiniLM‑L6‑v2`` | very fast, ~80 MB |
| ``distil‑bert`` | ``sentence‑transformers/distilbert‑base‑uncased`` | medium |
| ``xlm‑roberta`` | ``sentence‑transformers/paraphrase‑multilingual‑MPNET‑base‑v2`` | multilingual |

The ``get_embedding_function`` returns a callable that accepts a list of strings
and returns a ``list[np.ndarray]`` (or ``list[torch.Tensor]``) of embedding
vectors.  The function internally caches the model so repeated calls are cheap.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, List

try:
    # ``sentence_transformers`` pulls in ``torch`` – both are optional dependencies.
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The optional dependency 'sentence_transformers' is required for embedding "
        "functions. Install it via `pip install sentence-transformers`.") from e

# Mapping from a short identifier to the actual Hugging‑Face model name.
_MODEL_REGISTRY: dict[str, str] = {
    "bge-m3": "BAAI/bge-m3",
    "mini-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "distil-bert": "sentence-transformers/distilbert-base-uncased",
    "xlm-roberta": "sentence-transformers/paraphrase-multilingual-MPNET-base-v2",
}

def _load_model(model_name: str) -> SentenceTransformer:
    """Load a ``SentenceTransformer`` model given its registry key.

    Parameters
    ----------
    model_name: str
        The short identifier (e.g. ``"mini-l6"``).
    """
    hf_name = _MODEL_REGISTRY.get(model_name)
    if not hf_name:
        raise ValueError(f"Unsupported embedding model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")
    # ``SentenceTransformer`` will download the model on first use and cache it
    # under ``~/.cache/torch`` – this is free and only happens once per environment.
    return SentenceTransformer(hf_name)

# Cache the loaded model per identifier to avoid re‑loading on every call.
@functools.lru_cache(maxsize=None)
def _cached_model(model_name: str) -> SentenceTransformer:
    return _load_model(model_name)

def get_embedding_function(model_name: str | None = None) -> Callable[[List[str]], List]:
    """Return a function that converts a list of texts into embeddings.

    If *model_name* is ``None`` the function looks for the ``EMBEDDING_MODEL``
    environment variable; if that is also missing the default ``"mini-l6"`` is
    used (the fastest free model).

    The returned callable has the signature ``(texts: List[str]) -> List[np.ndarray]``.
    ``SentenceTransformer.encode`` is called with ``normalize_embeddings=False``
    to keep the raw vector values.
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "mini-l6")
    # Validate early so we can surface a clear error.
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Embedding model '{model_name}' is not recognised. Choose from: {list(_MODEL_REGISTRY)}"
        )
    model = _cached_model(model_name)

    def embed(texts: List[str]):
        # ``convert_to_numpy=True`` ensures we get ``np.ndarray`` objects.
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).tolist()

    return embed

def list_available_models() -> List[str]:
    """Return a list of the short identifiers for all supported models."""
    return list(_MODEL_REGISTRY.keys())

if __name__ == "__main__":  # pragma: no cover
    # Simple interactive demo when run directly.
    print("Available embedding models:")
    for key in sorted(_MODEL_REGISTRY):
        print(f" - {key}: {_MODEL_REGISTRY[key]}")
    choice = input("Select a model (default 'mini-l6'): ").strip() or "mini-l6"
    try:
        embed_fn = get_embedding_function(choice)
        sample = ["안녕하세요", "Hello world"]
        vectors = embed_fn(sample)
        print(f"Embedding vectors for {choice}: {vectors[:1]} ...")
    except Exception as exc:
        print(f"Error: {exc}")

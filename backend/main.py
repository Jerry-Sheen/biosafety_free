import os
"""
FastAPI Backend Core Server for Biosafety Free RAG.
Port: 8600 (Dedicated, avoids 8000/8501/8502 conflicts).
Provides SSE streaming chat, health checks, pathogen auto-completion, and ChromaDB search.
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.rag.vector_store import PersistentVectorStore
from src.rag.engine import ConversationalRAGEngine
from src.rag.local_llm import get_local_llm_provider
from src.security.logger import get_secure_logger

logger = get_secure_logger("backend_api")

# Global singleton resources
_vector_store: Optional[PersistentVectorStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes persistent vector store and warms up local embedding/LLM."""
    global _vector_store
    logger.info("🚀 Initializing Biosafety Free FastAPI Backend...")
    try:
        _vector_store = PersistentVectorStore()
        logger.info(f"✅ ChromaDB initialized at: {config.chroma_persist_dir}")
    except Exception as e:
        logger.warning(f"⚠️ ChromaDB initialization warning: {e}")

    # Warm up local LLM provider in a separate thread so startup is non-blocking
    def _warmup_llm():
        try:
            provider = get_local_llm_provider()
            logger.info("⏳ Background warming up local open-source LLM...")
            provider.load_model()
        except Exception as err:
            logger.warning(f"⚠️ Local LLM warmup deferred or failed: {err}")

    import threading
    warmup_thread = threading.Thread(target=_warmup_llm, daemon=True)
    warmup_thread.start()

    yield

    logger.info("🛑 Shutting down FastAPI Backend...")


app = FastAPI(
    title="Biosafety Free RAG Core API",
    version="1.0.0",
    description="Zero-cost, dual-mode (Free Open-Source & Paid BYOK) Biosafety Legal RAG Backend",
    lifespan=lifespan
)

# CORS Middleware (allows Streamlit frontend on port 8602/8601 to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Schemas
class ChatMessage(BaseModel):
    role: str = Field(description="Role: user or assistant")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or pathogen name")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Recent conversation turns")
    mode: str = Field(default="free", description="'free' (open-source CPU LLM) or 'paid' (BYOK cloud API)")
    provider: Optional[str] = Field(default="gemini", description="'gemini' or 'openai' (for paid mode)")
    api_key: Optional[str] = Field(default=None, description="API key provided by user in paid mode")
    model: Optional[str] = Field(default=None, description="Model identifier override")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4
    score_threshold: float = 0.0


@app.get("/api/v1/health")
async def health_check():
    """Returns the system health, model readiness, and storage statistics."""
    global _vector_store
    vs_count = 0
    if _vector_store:
        try:
            vs_count = _vector_store.count()
        except Exception:
            vs_count = -1

    local_llm = get_local_llm_provider()
    return {
        "status": "healthy",
        "app_env": config.app_env,
        "backend_port": config.backend_port,
        "chroma_chunks_count": vs_count,
        "local_embed_model": config.local_embed_model,
        "local_llm_model": config.local_llm_model,
        "local_llm_ready": local_llm.is_ready,
        "supported_modes": ["free", "paid"]
    }


@app.get("/api/v1/pathogen-keys")
async def list_pathogen_keys(source_name: str = "병원균(위험균) 종류 v3.json"):
    """Returns the list of pathogen keys indexed in ChromaDB for search autocompletion."""
    global _vector_store
    if not _vector_store:
        _vector_store = PersistentVectorStore()

    try:
        keys = _vector_store.get_pathogen_keys(source_name)
        return {"keys": keys, "count": len(keys)}
    except Exception as e:
        logger.error(f"Error fetching pathogen keys: {e}")
        return {"keys": [], "count": 0, "error": str(e)}


@app.post("/api/v1/search")
async def search_vector_store(req: SearchRequest):
    """Direct vector search endpoint using the unified sentence-transformers model."""
    global _vector_store
    if not _vector_store:
        _vector_store = PersistentVectorStore()

    try:
        results = _vector_store.search(
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold
        )
        return {"query": req.query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat")
async def chat_stream_endpoint(req: ChatRequest):
    """
    Main SSE Streaming Endpoint.
    Streams progress states, Step 1 classification, token chunks, citations, and completion.
    """
    global _vector_store
    if not _vector_store:
        _vector_store = PersistentVectorStore()

    # Determine effective LLM provider
    effective_provider = "free" if req.mode == "free" else (req.provider or "gemini").lower()

    # Instantiate engine with requested provider and keys
    gemini_key = req.api_key if effective_provider == "gemini" else config.gemini_api_key
    openai_key = req.api_key if effective_provider == "openai" else config.openai_api_key

    engine = ConversationalRAGEngine(
        vector_store=_vector_store,
        llm_provider=effective_provider,
        gemini_api_key=gemini_key,
        gemini_model=req.model if effective_provider == "gemini" else None,
        openai_api_key=openai_key,
        openai_model=req.model if effective_provider == "openai" else None
    )

    history_dicts = [{"role": m.role, "content": m.content} for m in req.chat_history]

    async def event_generator():
        try:
            # Run generator in thread pool since local inference is synchronous
            stream_gen = engine.generate_streaming_response(
                query=req.query,
                chat_history=history_dicts
            )

            for event in stream_gen:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Chat streaming error: {e}", exc_info=True)
            err_event = {
                "type": "error",
                "content": f"⚠️ 서버 처리 중 오류가 발생했습니다: {str(e)}"
            }
            yield f"data: {json.dumps(err_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=config.app_host,
        port=config.backend_port,
        reload=False
    )

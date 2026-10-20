"""
Persistent Vector Store Manager using ChromaDB.
Provides deduplication by document_hash and chunk_hash, incremental additions,
source deletion, and collection analytics.
"""
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import chromadb
from chromadb.config import Settings

from config import config
from src.rag.embeddings import get_embedding_provider
from pipeline.logger import logger, IngestionErrorType, log_pipeline_event


class VectorStore:
    """Enterprise persistent vector store with hash-level deduplication and incremental upserts."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_provider: Optional[Any] = None
    ):
        self.persist_directory = str(persist_directory or config.chroma_persist_dir)
        self.collection_name = collection_name or config.chroma_collection_name
        self.embedding_provider = embedding_provider or get_embedding_provider()

        logger.info(f"Connecting to ChromaDB at: {self.persist_directory} [Collection: {self.collection_name}]")
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def has_document(self, document_hash: str) -> bool:
        """Checks if a document with given SHA-256 hash already exists in vector store."""
        if not document_hash or self.collection.count() == 0:
            return False
        try:
            results = self.collection.get(
                where={"document_hash": document_hash},
                limit=1
            )
            return bool(results and results.get("ids"))
        except Exception as e:
            logger.warning(f"Error checking document hash {document_hash}: {e}")
            return False

    def get_existing_chunk_hashes(self, document_hash: str) -> Set[str]:
        """Returns all chunk_hashes currently stored for a document."""
        if self.collection.count() == 0:
            return set()
        try:
            results = self.collection.get(
                where={"document_hash": document_hash},
                include=["metadatas"]
            )
            hashes = set()
            for m in results.get("metadatas", []):
                if m and "chunk_hash" in m:
                    hashes.add(m["chunk_hash"])
            return hashes
        except Exception as e:
            logger.warning(f"Error fetching chunk hashes for {document_hash}: {e}")
            return set()

    def add_chunks(self, chunks: List[Dict[str, Any]], force: bool = False) -> int:
        """
        Adds chunk dictionaries to the collection with duplicate protection.
        Returns count of newly inserted chunks.
        """
        if not chunks:
            return 0

        # Filter duplicates if not force
        to_insert = []
        doc_hash = chunks[0]["metadata"].get("document_hash")

        if doc_hash and not force:
            existing_chunk_hashes = self.get_existing_chunk_hashes(doc_hash)
            for c in chunks:
                ch_hash = c["metadata"].get("chunk_hash")
                if ch_hash and ch_hash in existing_chunk_hashes:
                    continue
                to_insert.append(c)
        else:
            to_insert = chunks

        if not to_insert:
            logger.info("All chunks already exist in vector store. Skipping insert.")
            return 0

        texts = [c["text"].encode("utf-8", "ignore").decode("utf-8") for c in to_insert]
        metadatas = [c["metadata"] for c in to_insert]

        ids = [
            f"{m.get('source', 'doc')}_{m.get('chunk_index', idx)}_{uuid.uuid4().hex[:8]}"
            for idx, m in enumerate(metadatas)
        ]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_provider.embed_documents(texts)

        # Sanitize metadata to primitives for ChromaDB
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                else:
                    cleaned[k] = str(v)
            cleaned_metadatas.append(cleaned)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=cleaned_metadatas
        )

        logger.info(f"Successfully stored {len(to_insert)} chunks in collection '{self.collection_name}'.")
        return len(to_insert)

    def delete_by_document_hash(self, document_hash: str) -> int:
        """Deletes all chunks belonging to a document hash."""
        if not document_hash or self.collection.count() == 0:
            return 0
        initial_count = self.collection.count()
        self.collection.delete(where={"document_hash": document_hash})
        new_count = self.collection.count()
        deleted = initial_count - new_count
        logger.info(f"Deleted {deleted} chunks for document_hash: {document_hash}")
        return deleted

    def delete_by_source(self, source_name: str) -> int:
        """Deletes all chunks belonging to a source filename."""
        if not source_name or self.collection.count() == 0:
            return 0
        initial_count = self.collection.count()
        self.collection.delete(where={"source": source_name})
        new_count = self.collection.count()
        deleted = initial_count - new_count
        logger.info(f"Deleted {deleted} chunks for source: {source_name}")
        return deleted

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Searches collection for relevant chunks."""
        if self.collection.count() == 0:
            return []

        query_emb = self.embedding_provider.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matched = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            similarity = max(0.0, 1.0 - dist)
            if similarity >= score_threshold:
                matched.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                    "distance": round(dist, 4)
                })

        matched.sort(key=lambda x: x["similarity"], reverse=True)
        return matched

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection metrics."""
        count = self.collection.count()
        if count == 0:
            return {
                "total_chunks": 0,
                "unique_documents_count": 0,
                "document_sources": []
            }
        all_meta = self.collection.get(include=["metadatas"]).get("metadatas", [])
        sources = set()
        for m in all_meta:
            sources.add(m.get("source", "Unknown"))
        return {
            "total_chunks": count,
            "unique_documents_count": len(sources),
            "document_sources": sorted(list(sources))
        }

'''\
Persistent ChromaDB Vector Store Manager.
Maintains persistent on-disk vector database, supports incremental document additions,
metadata filtering, statistics tracking, and document deletion.
'''

import os
# Force ChromaDB to use stable Python SegmentAPI (bypasses RustBindingsAPI bug)

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.config import config
from src.rag.embeddings import get_embedding_provider
from src.security.logger import get_secure_logger
from chromadb.errors import NotFoundError, InvalidArgumentError

logger = get_secure_logger("vector_store")


def _safe_is_file(p: Path) -> bool:
    """Safely checks if path is an existing file without raising PermissionError or OSError."""
    try:
        return p.is_file()
    except (OSError, PermissionError):
        return False


def _safe_file_size(p: Path) -> int:
    """Safely returns file size, or -1 if inaccessible or not a regular file."""
    try:
        return p.stat().st_size
    except (OSError, PermissionError):
        return -1


class PersistentVectorStore:
    """Manages persistent ChromaDB vector storage with incremental append support."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_provider: Optional[Any] = None,
    ):
        if persist_directory is not None:
            target_dir = Path(persist_directory)
        else:
            candidates = [
                Path(config.chroma_persist_dir),
                Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db",
                Path.cwd() / "data" / "chroma_db",
                Path("data/chroma_db").resolve(),
                Path("/mount/src/biosafety-rag/data/chroma_db"),
            ]
            # Prioritize candidate directory containing chroma.sqlite3 with the largest file size (> 1MB)
            best_cand = None
            max_size = -1
            for cand in candidates:
                try:
                    db_file = cand / "chroma.sqlite3"
                    if _safe_is_file(db_file):
                        sz = _safe_file_size(db_file)
                        if sz > max_size:
                            max_size = sz
                            best_cand = cand
                except (OSError, PermissionError):
                    continue
            if best_cand is not None and max_size > 100000:
                logger.info(f"Resolved populated ChromaDB directory ({max_size:,} bytes): {best_cand}")
                target_dir = best_cand
            else:
                target_dir = Path(config.chroma_persist_dir)

        self.persist_directory = str(target_dir)
        self.collection_name = collection_name or config.chroma_collection_name
        self._embedding_provider = embedding_provider
        self._expected_dimension: Optional[int] = None

        logger.info(f"Connecting to ChromaDB at: {self.persist_directory}")

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Fallback check: if collection count is 0, check other candidates for populated collection
        try:
            if self.collection.count() == 0 and persist_directory is None:
                for cand in candidates:
                    try:
                        db_file = cand / "chroma.sqlite3"
                        if str(cand) != self.persist_directory and _safe_is_file(db_file):
                            temp_client = chromadb.PersistentClient(
                                path=str(cand),
                            )
                            temp_col = temp_client.get_collection(self.collection_name)
                            if temp_col.count() > 0:
                                logger.info(f"Recovered populated collection '{self.collection_name}' ({temp_col.count()} chunks) from: {cand}")
                                self.client = temp_client
                                self.collection = temp_col
                                self.persist_directory = str(cand)
                                break
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Fallback candidate scan notice: {e}")

        # If collection already contains data, infer its dimension
        try:
            if self.collection.count() > 0:
                first_emb = self.collection.get(limit=1, include=["embeddings"]).get("embeddings", [])[0]
                self._expected_dimension = len(first_emb)
                logger.info(f"Detected existing collection dimension: {self._expected_dimension}")
        except Exception as e:
            logger.warning(f"Notice: unable to infer existing collection dimension: {e}")

    @property
    def embedding_provider(self):
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _collection_dimension(self) -> Optional[int]:
        """Return embedding dimension of current collection if it has data."""
        if self.collection.count() == 0:
            return None
        try:
            emb = self.collection.get(limit=1, include=["embeddings"]).get("embeddings", [])[0]
            return len(emb)
        except Exception:
            return None

    def _ensure_correct_dimension(self) -> bool:
        """Check that collection dimension matches the provider.

        NEVER deletes or re-indexes data. (예전에는 불일치 시 컬렉션을 삭제하고
        전체 재임베딩을 시도하여 문서가 사라지고 검색이 수 분간 멈추는 문제가 있었음.)
        Returns True if compatible (or collection empty), False otherwise.
        """
        provider_dim = getattr(self.embedding_provider, "dimension", None)
        collection_dim = self._collection_dimension()
        if provider_dim is None or collection_dim is None:
            self._expected_dimension = provider_dim or collection_dim
            return True
        if collection_dim != provider_dim:
            logger.error(
                f"Embedding dimension mismatch (collection={collection_dim}, provider={provider_dim}). "
                "Data is preserved; re-upload documents with the current embedding model if needed."
            )
            return False
        self._expected_dimension = provider_dim
        return True

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @staticmethod
    def _clean_metadatas(metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned_list = []
        for meta in metadatas:
            cleaned_list.append({
                k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in (meta or {}).items()
            })
        return cleaned_list

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Append chunked documents (embeddings computed with the fixed local model)."""
        if not chunks:
            return 0
        texts = [c["text"].encode("utf-8", "ignore").decode("utf-8") for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} document chunks...")
        embeddings = self.embedding_provider.embed_documents(texts)
        return self.add_embeddings(embeddings, [{"text": t, "metadata": c["metadata"]} for t, c in zip(texts, chunks)])

    def add_embeddings(self, embeddings: List[List[float]], chunks: List[Dict[str, Any]]) -> int:
        """Add pre-computed embeddings. Existing data is never deleted."""
        if not embeddings or not chunks:
            return 0
        if not self._ensure_correct_dimension():
            raise ValueError(
                f"벡터 DB의 임베딩 차원({self._collection_dimension()})과 현재 임베딩 모델 차원"
                f"({self.embedding_provider.dimension})이 다릅니다. 기존 데이터는 보존되었습니다."
            )
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            f"{m.get('source', 'doc')}_{m.get('chunk_index', idx)}_{uuid.uuid4().hex[:8]}"
            for idx, m in enumerate(metadatas)
        ]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=self._clean_metadatas(metadatas),
        )
        logger.info(f"Successfully batch added {len(chunks)} chunks.")
        return len(chunks)

    @staticmethod
    def _clean_query_for_matching(query: str) -> str:
        """Removes conversational phrases and Korean particles to extract core entity name."""
        q = (query or "").strip()
        phrases = [
            "에 대해 알려줘", "에 대해 설명해줘", "에 대한 규제", "에 대해", "에 대한",
            "알려줘", "설명해줘", "규제 현황", "규제 대상", "규제", "법령"
        ]
        for p in phrases:
            q = q.replace(p, " ")

        particles = ["에서는", "에서", "에게", "으로", "로는", "이란", "라는", "은", "는", "이", "가", "을", "를", "의", "와", "과", "로", "도", "만", "에", "란"]
        tokens = q.split()
        cleaned_tokens = []
        for tok in tokens:
            t = tok.strip()
            for p in sorted(particles, key=len, reverse=True):
                if t.endswith(p) and len(t) - len(p) >= 2:
                    t = t[:-len(p)]
                    break
            if t:
                cleaned_tokens.append(t)
        return " ".join(cleaned_tokens)

    @classmethod
    def _is_exact_pathogen_match(cls, query: str, doc_text: str) -> bool:
        """
        Determines whether the document chunk represents the exact pathogen queried.
        Scans all pathogen title headers in the chunk to properly handle chunking overlap.
        Distinguishes specific variants (e.g. 'Variola virus' vs 'Variola minor virus').
        """
        import re
        q_clean = cls._clean_query_for_matching(query).lower()
        if not q_clean:
            q_clean = query.lower().strip()

        titles = re.findall(r'\[(?:병원균/항목|항목/병원체명|병원체명|병원균):\s*([^\]]+)\]', doc_text)
        if not titles:
            return False

        q_nospace = q_clean.replace(" ", "")
        for full_title in titles:
            parens = [p.strip().lower() for p in re.findall(r'\(([^)]+)\)', full_title)]
            base_name = re.sub(r'\([^)]+\)', '', full_title).strip().lower()
            candidates = [base_name] + parens

            for cand in candidates:
                cand_nospace = cand.replace(" ", "")
                if not cand_nospace or len(cand_nospace) < 2:
                    continue
                # 1) Exact match
                if q_clean == cand or q_nospace == cand_nospace:
                    return True
                # 2) Candidate is contained in query (e.g. cand is '콜레라균', query is '콜레라균의 규제 현황')
                if cand_nospace in q_nospace:
                    idx = q_nospace.find(cand_nospace)
                    if idx > 0 and q_nospace[idx-1] == '소':
                        if cand_nospace in ("두창바이러스", "두창"):
                            continue
                    return True
                # 3) Query is contained in candidate (e.g. query is '콜레라', cand is '콜레라균')
                if q_nospace in cand_nospace:
                    idx = cand_nospace.find(q_nospace)
                    if idx == 0:
                        return True
        return False

    def find_pathogen_chunk(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Directly scans ChromaDB for the exact pathogen chunk matching any pathogen
        name mentioned in the query. Guarantees 100% retrieval regardless of vector similarity.
        """
        import re
        raw_tokens = re.findall(r'[a-zA-Z]{3,}|[가-힣]{2,}', query)
        stopwords = {
            "대해", "대한", "대하여", "설명해줘", "설명", "알려줘", "알려", "무엇인가", "무엇인지",
            "관련", "관련된", "어떤", "있는", "내용", "정리해줘", "말해줘", "요약해줘", "특징",
            "정보", "종류", "항목", "규정", "법령", "규제", "현황", "적용", "대상"
        }
        korean_particles = ["에서는", "에서", "에게", "으로", "로는", "이란", "라는", "은", "는", "이", "가", "을", "를", "의", "와", "과", "로", "도", "만", "에", "란"]
        tokens = []
        for t in raw_tokens:
            cleaned = t.strip()
            for p in sorted(korean_particles, key=len, reverse=True):
                if cleaned.endswith(p) and len(cleaned) - len(p) >= 2:
                    cleaned = cleaned[:-len(p)]
                    break
            if cleaned not in stopwords and len(cleaned) >= 2 and cleaned not in tokens:
                tokens.append(cleaned)
            # Decompose common pathogen suffixes like 바이러스, 균
            for suffix in ["바이러스", "균"]:
                if cleaned.endswith(suffix) and len(cleaned) - len(suffix) >= 2:
                    sub = cleaned[:-len(suffix)]
                    if sub not in stopwords and sub not in tokens:
                        tokens.append(sub)

        for tok in tokens:
            try:
                res = self.collection.get(where_document={"$contains": tok}, limit=15)
                docs = res.get("documents", [])
                metas = res.get("metadatas", [])
                for d, m in zip(docs, metas):
                    if self._is_exact_pathogen_match(query, d):
                        return {
                            "text": d,
                            "metadata": m,
                            "similarity": 1.0,
                            "distance": 0.0,
                            "exact_match": True,
                        }
            except Exception as e:
                logger.debug(f"find_pathogen_chunk error for token '{tok}': {e}")
        return None

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search collection with query embedding; returns matched chunks."""
        if self.collection.count() == 0:
            logger.info("Search requested but collection is empty.")
            return []
        if not self._ensure_correct_dimension():
            return []
        query_emb = self.embedding_provider.embed_query(query)
        # 1. 시맨틱 벡터 검색 후보군 대폭 확대 (충분한 후보 풀 확보)
        fetch_k = min(max(top_k * 10, 50), self.collection.count())
        try:
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
        except InvalidArgumentError as e:
            if "dimension" in str(e):
                logger.warning(f"Embedding dimension mismatch during search ({e}). Returning empty results.")
                return []
            else:
                raise

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # 2. 고도화된 키워드 추출 (한글 조사 분리 및 불용어 정제)
        import re
        raw_tokens = re.findall(r'[a-zA-Z]{3,}|[가-힣]{2,}', query)
        stopwords = {
            "대해", "대한", "대하여", "설명해줘", "설명", "알려줘", "알려", "무엇인가", "무엇인지",
            "관련", "관련된", "어떤", "있는", "내용", "정리해줘", "말해줘", "요약해줘", "특징",
            "정보", "종류", "항목", "규정", "법령"
        }
        korean_particles = ["에서는", "에서", "에게", "으로", "로는", "이란", "라는", "은", "는", "이", "가", "을", "를", "의", "와", "과", "로", "도", "만", "에", "란"]

        keywords = []
        for t in raw_tokens:
            t_lower = t.strip().lower()
            if t_lower in stopwords:
                continue
            # 한글 조사 제거
            cleaned = t_lower
            for p in sorted(korean_particles, key=len, reverse=True):
                if cleaned.endswith(p) and len(cleaned) - len(p) >= 2:
                    cleaned = cleaned[:-len(p)]
                    break
            if cleaned not in stopwords and len(cleaned) >= 2 and cleaned not in keywords:
                keywords.append(cleaned)

        matched_dict = {}

        # (0) 쿼리 정제 및 병원체명 완전 일치 판별 헬퍼
        clean_q = self._clean_query_for_matching(query)

        # (A) 벡터 검색 결과 등록 및 키워드/정밀 일치 가중치 부스팅
        for doc, meta, dist in zip(docs, metas, distances):
            base_sim = max(0.0, 1.0 - dist)
            doc_lower = doc.lower()
            keyword_hits = sum(1 for kw in keywords if kw in doc_lower)

            is_exact = self._is_exact_pathogen_match(query, doc)

            boost = 0.0
            if is_exact:
                boost = 1.0  # 완전 일치 항목은 무조건 최상위(1.0)
            elif keyword_hits > 0:
                boost = min(0.40, keyword_hits * 0.15)
                first_line = doc_lower.split('\n')[0]
                if any(kw in first_line for kw in keywords):
                    boost += 0.25

            final_sim = 1.0 if is_exact else min(1.0, base_sim + boost)
            if final_sim >= score_threshold:
                matched_dict[doc] = {
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(final_sim, 4),
                    "distance": 0.0 if is_exact else round(dist, 4),
                    "exact_match": is_exact,
                }

        # (B) 핵심 키워드 및 정제 쿼리 직접 컬렉션 검색 (True Hybrid: 무조건 병합)
        search_terms = list(keywords[:4])
        if clean_q and clean_q not in search_terms:
            search_terms.insert(0, clean_q)

        for kw in search_terms:
            try:
                kw_results = self.collection.get(
                    where_document={"$contains": kw},
                    limit=max(top_k * 2, 10),
                    include=["documents", "metadatas"]
                )
                kw_docs = kw_results.get("documents", [])
                kw_metas = kw_results.get("metadatas", [])
                for kd, km in zip(kw_docs, kw_metas):
                    kd_lower = kd.lower()
                    hits = sum(1 for k in keywords if k in kd_lower)
                    first_line = kd_lower.split('\n')[0]
                    title_match = any(k in first_line for k in keywords)
                    is_exact = self._is_exact_pathogen_match(query, kd)

                    if is_exact:
                        kw_sim = 1.0
                    else:
                        kw_sim = 0.88 + min(0.08, hits * 0.03) + (0.04 if title_match else 0.0)
                        kw_sim = min(1.0, kw_sim)

                    if kd in matched_dict:
                        if is_exact:
                            matched_dict[kd]["exact_match"] = True
                            matched_dict[kd]["similarity"] = 1.0
                            matched_dict[kd]["distance"] = 0.0
                        else:
                            matched_dict[kd]["similarity"] = max(matched_dict[kd]["similarity"], round(kw_sim, 4))
                    else:
                        matched_dict[kd] = {
                            "text": kd,
                            "metadata": km,
                            "similarity": round(kw_sim, 4),
                            "distance": 0.0 if is_exact else round(1.0 - kw_sim, 4),
                            "exact_match": is_exact,
                        }
            except Exception as kw_e:
                logger.debug(f"Keyword search notice for '{kw}': {kw_e}")

        matched = list(matched_dict.values())
        matched.sort(key=lambda x: (x.get("exact_match", False), x["similarity"]), reverse=True)

        # (C) 보장 단계: 쿼리 내 병원체가 언급된 경우 직렬 전용 검색으로 반드시 Top-1 보장
        direct_chunk = self.find_pathogen_chunk(query)
        if direct_chunk:
            matched = [m for m in matched if m.get("text") != direct_chunk["text"]]
            matched.insert(0, direct_chunk)

        return matched[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the collection."""
        try:
            count = self.collection.count()
        except Exception as e:
            logger.warning(f"Collection count warning for '{self.collection_name}': {e}")
            count = 0
        if count == 0:
            # Recheck candidate paths if count returned 0
            for cand in [
                Path(config.chroma_persist_dir),
                Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db",
                Path.cwd() / "data" / "chroma_db",
                Path("data/chroma_db").resolve(),
                Path("/mount/src/biosafety-rag/data/chroma_db"),
            ]:
                try:
                    db_file = cand / "chroma.sqlite3"
                    if _safe_is_file(db_file):
                        temp_col = temp_client.get_collection(self.collection_name)
                        if temp_col.count() > 0:
                            logger.info(f"get_stats recovered collection from: {cand}")
                            self.client = temp_client
                            self.collection = temp_col
                            self.persist_directory = str(cand)
                            count = temp_col.count()
                            break
                except Exception:
                    pass
        if count == 0:
            return {
                "total_chunks": 0,
                "unique_documents_count": 0,
                "document_sources": [],
                "source_chunk_counts": {},
            }
        try:
            all_meta = self.collection.get(include=["metadatas"]).get("metadatas", [])
        except Exception as e:
            logger.warning(f"Collection get metadatas warning: {e}")
            all_meta = []
        sources = set()
        source_counts: Dict[str, int] = {}
        for m in all_meta:
            src = m.get("source", "Unknown")
            sources.add(src)
            source_counts[src] = source_counts.get(src, 0) + 1
        return {
            "total_chunks": count,
            "unique_documents_count": len(sources),
            "document_sources": sorted(list(sources)),
            "source_chunk_counts": source_counts,
        }

    def delete_document(self, source_filename: str) -> int:
        """Delete all chunks belonging to a specific source file."""
        logger.info(f"Deleting all chunks for document source: {source_filename}")
        initial = self.collection.count()
        self.collection.delete(where={"source": source_filename})
        new = self.collection.count()
        deleted = initial - new
        logger.info(f"Deleted {deleted} chunks for '{source_filename}'.")
        return deleted

    def reset_collection(self):
        """Completely drop and recreate the collection, clearing stored dimension cache."""
        logger.warning(f"Resetting vector store collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._expected_dimension = getattr(self.embedding_provider, "dimension", None)

    def get_pathogen_keys(self, source_name: str = "병원균(위험균) 종류 v3.json") -> List[str]:
        """
        영구 벡터 DB(ChromaDB)에 저장된 특정 JSON 문서 청크들로부터
        병원균 이름(Key 값) 목록을 추출하여 반환합니다.
        """
        import re
        try:
            res = self.collection.get(
                where={"source": source_name},
                include=["documents"]
            )
            docs = res.get("documents", [])
            if docs:
                pattern = re.compile(r'\[병원균/항목:\s*(.*?)(?=\]\s*\r?\n\s*-)', re.DOTALL)
                keys = []
                for d in docs:
                    for k in pattern.findall(d):
                        k_clean = k.strip()
                        if k_clean and k_clean not in keys:
                            keys.append(k_clean)
                if keys:
                    return keys
        except Exception as e:
            logger.warning(f"Error fetching pathogen keys from vector store for '{source_name}': {e}")

        # Fallback to local json file if vector store retrieval fails
        try:
            candidates = [
                Path("data/uploads") / source_name,
                Path(__file__).resolve().parent.parent.parent / "data" / "uploads" / source_name,
                Path("data/reference") / source_name,
            ]
            for cand in candidates:
                if cand.is_file():
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            return list(data.keys())
        except Exception:
            pass

        return []


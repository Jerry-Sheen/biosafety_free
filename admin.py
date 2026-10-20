import os
"""
==============================================================================
Admin Application (admin.py)
------------------------------------------------------------------------------
관리자 전용 문서 관리 및 영구 벡터 DB 구축 시스템.
독립적으로 실행 가능: streamlit run admin.py

기능:
1. PDF, EXCEL, Word, CSV, HWP/HWPX 등 다양한 문서 다중 동시 업로드.
2. 스캔 PDF 이미지 OCR 및 Docling 기반 구조화(텍스트, 표, 메타데이터) 변환.
3. 문서 청킹 및 로컬 임베딩(SentenceTransformers, user.py와 동일 모델 고정) 후 영구 벡터 DB(ChromaDB)에 저장. (Ollama 지원 제거됨)
4. 증분 저장(Incremental Ingestion): 기존 데이터를 덮어쓰지 않고 새로운 문서 계속 추가.
5. 등록 문서 현황 조회, 개별 문서 삭제, 전체 DB 초기화 지원.
==============================================================================
"""
from pathlib import Path
import streamlit as st

from src.config import config
from src.security.secrets_manager import secrets_manager
from src.parsers.document_parser import DocumentParser
from src.rag.chunker import DocumentChunker
from src.rag.vector_store import PersistentVectorStore
from src.security.logger import get_secure_logger

logger = get_secure_logger("admin_app")

# Streamlit Page Settings (Safe for both standalone and imported execution)
try:
    st.set_page_config(
        page_title="RAG Vibe - 관리자 문서 관리 시스템 (Admin)",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

# Custom Styling
st.markdown(
    """
    <style>
    .admin-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .admin-desc {
        color: #4B5563;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="벡터 데이터베이스 초기화 중...")
def get_vector_store() -> PersistentVectorStore:
    """Initializes and caches the persistent vector store."""
    return PersistentVectorStore()


def main():
    st.markdown('<div class="admin-title">🛠️ RAG Vibe - 관리자 문서 관리 포털</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="admin-desc">문서를 업로드하여 Docling/OCR로 파싱하고, 로컬 임베딩 모델을 통해 영구 벡터 DB에 증분 저장합니다.</div>',
        unsafe_allow_html=True
    )

    # Initialize vector store (fixed local embedding model shared with user.py; Ollama removed)
    vector_store = get_vector_store()


    # 1. Admin PIN Verification
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        with st.container(border=True):
            st.subheader("🔒 관리자 보안 인증")
            pin_input = st.text_input("관리자 PIN 번호를 입력하세요:", type="password")
            if st.button("인증하기", type="primary"):
                if pin_input == config.admin_access_pin:
                    st.session_state.admin_authenticated = True
                    st.success("관리자 인증에 성공했습니다.")
                    st.rerun()
                else:
                    st.error("PIN 번호가 올바르지 않습니다.")
            st.caption(f"기본 PIN은 `.env`의 `ADMIN_ACCESS_PIN`에 설정되어 있습니다 (기본: {config.admin_access_pin})")
        return

    # 2. Sidebar - System & Embedding Status
    with st.sidebar:
        st.subheader("🖥️ 시스템 상태")
        st.markdown(f"**환경:** `{config.app_env.upper()}`")
        st.markdown(f"**영구 DB 경로:** `{config.chroma_persist_dir.name}`")

        st.divider()
        st.subheader("🔧 임베딩 모델 (고정)")
        st.info(
            f"`{vector_store.embedding_provider.model_name}`\n\n"
            f"차원: {vector_store.embedding_provider.dimension} · 로컬 실행(무료)\n\n"
            "user.py와 동일한 모델을 사용하므로 새로고침/재시작 시에도 데이터가 유지됩니다."
        )

        st.divider()
        if st.button("🚪 관리자 로그아웃", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()

    # 3. Main Tabs
    tab_upload, tab_db = st.tabs(["📤 문서 업로드 및 벡터 DB 색인", "📊 영구 벡터 DB 현황 및 관리"])

    # ------------------ TAB 1: UPLOAD & INGEST ------------------
    with tab_upload:
        st.subheader("1. 문서 파일 업로드 (다중 파일 동시 선택 가능)")
        st.write("지원 포맷: **PDF, Word(`.docx`), Excel(`.xlsx`, `.xls`), CSV, HWP, HWPX, Markdown, TXT**")

        uploaded_files = st.file_uploader(
            "문서를 선택하거나 여기에 드래그 앤 드롭하세요:",
            type=["pdf", "docx", "doc", "xlsx", "xls", "csv", "hwp", "hwpx", "txt", "md", "json"],
            accept_multiple_files=True
        )

        st.divider()
        st.subheader("2. 파싱 및 청킹 파라미터")
        col1, col2 = st.columns(2)
        with col1:
            use_docling = st.checkbox("Docling AI 구조화 파서 사용 (추천)", value=True, help="문서 레이아웃 및 복잡한 표를 마크다운 구조로 보존합니다.")
            use_ocr = st.checkbox("스캔 PDF 이미지 OCR 활성화", value=True, help="텍스트 레이어가 없는 이미지 기반 PDF에서 글자를 추출합니다.")

        with col2:
            chunk_size = st.slider("청크 크기 (Chunk Size)", min_value=300, max_value=2000, value=800, step=50)
            chunk_overlap = st.slider("청크 중복 크기 (Chunk Overlap)", min_value=50, max_value=400, value=150, step=25)

        # Ingestion Button
        if st.button("🚀 문서 파싱 및 벡터 DB 추가 (Incremental Ingestion)", type="primary", disabled=not uploaded_files):
            # Initialize progress placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            percent_text = st.empty()

            parser = DocumentParser(prefer_docling=use_docling, enable_ocr=use_ocr)
            chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            total_files = len(uploaded_files)
            total_steps = total_files * 2  # parsing + embedding for each file
            completed_steps = 0
            total_added_chunks = 0
            success_files = []
            failed_files = []

            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name

                # Step 1: Parsing
                status_text.text(f"[{idx+1}/{total_files}] 파싱 중: {filename}...")
                # Save file to upload directory
                save_path = config.upload_dir / filename
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                parse_result = parser.parse_file(save_path)
                if not parse_result.get("success", False):
                    failed_files.append((filename, parse_result.get("error", "Unknown parse error")))
                    completed_steps += 1
                    percent = int(completed_steps / total_steps * 100)
                    percent_text.text(f"진행률: {percent}%")
                    progress_bar.progress(completed_steps / total_steps)
                    continue

                completed_steps += 1
                percent = int(completed_steps / total_steps * 100)
                percent_text.text(f"진행률: {percent}%")
                progress_bar.progress(completed_steps / total_steps)

                                # Step 2: Chunking & Embedding (batch)
                doc_text = parse_result.get("text", "")
                metadata = parse_result.get("metadata", {})
                chunks = chunker.chunk_document(doc_text, metadata)
                # 동일 파일 재업로드 시 기존 청크 삭제 후 최신 내용으로 갱신
                vector_store.delete_document(filename)
                status_text.text(f"[{idx+1}/{total_files}] 임베딩 생성 및 영구 저장 중: {filename} ({len(chunks)}개 청크)...")
                # Prepare texts for batch embedding
                texts = [c["text"] for c in chunks]
                try:
                    embeddings = vector_store.embedding_provider.embed_documents(texts)
                    added_count = vector_store.add_embeddings(embeddings, chunks)
                except Exception as e:
                    logger.error(f"Embedding/storage failed for {filename}: {e}")
                    failed_files.append((filename, f"임베딩/저장 실패: {e}"))
                    completed_steps += 1
                    progress_bar.progress(completed_steps / total_steps)
                    continue
                total_added_chunks += added_count
                success_files.append((filename, added_count))

                completed_steps += 1
                percent = int(completed_steps / total_steps * 100)
                percent_text.text(f"진행률: {percent}%")
                progress_bar.progress(completed_steps / total_steps)

            # Finalize UI
            percent_text.text("진행률: 100%")
            status_text.empty()
            progress_bar.progress(1.0)
            st.success(f"총 {len(success_files)}개 문서에서 {total_added_chunks}개 청크가 벡터 DB에 성공적으로 저장되었습니다!")

            if success_files:
                with st.expander("✅ 성공적으로 색인된 문서 목록", expanded=True):
                    for fname, count in success_files:
                        st.markdown(f"- 📄 **{fname}**: {count}개 청크 저장 완료")

            if failed_files:
                with st.expander("⚠️ 파싱 실패 문서 목록", expanded=True):
                    for fname, err in failed_files:
                        st.error(f"- **{fname}**: {err}")


    # ------------------ TAB 2: DB STATS & MANAGEMENT ------------------
    with tab_db:
        st.subheader("📊 영구 벡터 데이터베이스 현황")
        stats = vector_store.get_stats()

        col_st1, col_st2 = st.columns(2)
        col_st1.metric("총 임베딩 청크 수", f"{stats['total_chunks']:,} 개")
        col_st2.metric("색인된 고유 문서 수", f"{stats['unique_documents_count']:,} 개")

        st.divider()
        st.subheader("📁 색인된 개별 문서 관리")

        if stats["unique_documents_count"] == 0:
            st.info("현재 영구 벡터 DB에 저장된 문서가 없습니다. '문서 업로드' 탭에서 문서를 추가해주세요.")
        else:
            doc_counts = stats.get("source_chunk_counts", {})
            for doc_name, count in doc_counts.items():
                col_d1, col_d2 = st.columns([8, 2])
                with col_d1:
                    st.markdown(f"📄 **{doc_name}** (`{count}`개 청크)")
                with col_d2:
                    if st.button("문서 삭제", key=f"del_{doc_name}", type="secondary"):
                        deleted = vector_store.delete_document(doc_name)
                        st.warning(f"'{doc_name}' 관련 {deleted}개 청크가 삭제되었습니다.")
                        st.rerun()

            st.divider()
            with st.expander("⚠️ 위험 작업: 벡터 DB 전체 초기화", expanded=False):
                st.write("저장된 모든 문서 임베딩이 삭제되며 복구할 수 없습니다.")
                confirm = st.checkbox("네, 모든 데이터를 삭제하고 초기화하는 것에 동의합니다.")
                if st.button("전체 벡터 DB 초기화", type="primary", disabled=not confirm):
                    vector_store.reset_collection()
                    st.success("벡터 DB 컬렉션이 초기화되었습니다.")
                    st.rerun()


if __name__ == "__main__":
    main()

"""
Admin UI View for Streamlit.
Provides document uploading, parsing options (Docling, OCR), chunking settings,
incremental vector DB ingestion, collection statistics, and document deletion.
"""
from pathlib import Path
import streamlit as st
from src.config import config
from src.parsers.document_parser import DocumentParser
from src.rag.chunker import DocumentChunker
from src.rag.vector_store import PersistentVectorStore
from src.security.logger import get_secure_logger

logger = get_secure_logger("admin_view")


def render_admin_view(vector_store: PersistentVectorStore):
    """Renders the Admin administration panel."""
    st.header("🛠️ 관리자 패널 (Admin Panel)")
    st.caption("문서 업로드, Docling/OCR 변환, 청킹 설정 및 영구 벡터 DB를 관리합니다.")

    # 1. Admin PIN Verification (Optional security layer)
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        with st.container(border=True):
            st.subheader("🔒 관리자 인증")
            pin_input = st.text_input("관리자 PIN 번호를 입력하세요:", type="password")
            if st.button("인증하기", type="primary"):
                if pin_input == config.admin_access_pin:
                    st.session_state.admin_authenticated = True
                    st.success("인증에 성공했습니다.")
                    st.rerun()
                else:
                    st.error("PIN 번호가 일치하지 않습니다.")
            st.info("기본 테스트 PIN: `.env` 파일의 `ADMIN_ACCESS_PIN` (기본값: admin1234)")
        return

    # Logout button
    col_auth1, col_auth2 = st.columns([8, 2])
    with col_auth2:
        if st.button("🚪 관리자 로그아웃"):
            st.session_state.admin_authenticated = False
            st.rerun()

    # System Status & Security Badge
    with st.expander("🛡️ 시스템 보안 및 환경 상태", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("애플리케이션 환경", config.app_env.upper())
        col_s2.metric("LLM 모델", config.gemini_model)
        col_s3.metric("영구 DB 경로", str(config.chroma_persist_dir.name))

    tabs = st.tabs(["📤 문서 업로드 및 임베딩", "📊 벡터 DB 현황 및 관리"])

    # ------------------ TAB 1: UPLOAD & INGEST ------------------
    with tabs[0]:
        st.subheader("1. 문서 파일 업로드 (다중 선택 가능)")
        st.write("PDF, Word(.docx), Excel(.xlsx), CSV, HWP, HWPX, Markdown 문서를 지원합니다.")

        uploaded_files = st.file_uploader(
            "문서를 드래그 앤 드롭하거나 선택하세요:",
            type=["pdf", "docx", "doc", "xlsx", "xls", "csv", "hwp", "hwpx", "txt", "md"],
            accept_multiple_files=True
        )

        st.divider()
        st.subheader("2. 파싱 및 청킹 파라미터 설정")

        col1, col2 = st.columns(2)
        with col1:
            enable_docling = st.checkbox("Docling 구조화 파서 사용 (추천)", value=True, help="표와 문서 레이아웃을 마크다운 구조로 보존합니다.")
            enable_ocr = st.checkbox("PDF 이미지 OCR 활성화", value=True, help="스캔된 이미지 PDF에서 글자를 추출합니다.")

        with col2:
            chunk_size = st.slider("청크 크기 (Chunk Size)", min_value=300, max_value=2000, value=800, step=50)
            chunk_overlap = st.slider("청크 중복 (Chunk Overlap)", min_value=50, max_value=400, value=150, step=25)

        if st.button("🚀 문서 파싱 및 벡터 DB 추가 (Incremental Ingestion)", type="primary", disabled=not uploaded_files):
            progress_bar = st.progress(0)
            status_text = st.empty()

            parser = DocumentParser(prefer_docling=enable_docling, enable_ocr=enable_ocr)
            chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            total_files = len(uploaded_files)
            total_added_chunks = 0
            success_files = []
            failed_files = []

            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.text(f"[{idx+1}/{total_files}] 처리 중: {filename}...")

                # Save file to uploads directory temporarily
                save_path = config.upload_dir / filename
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Parse document
                parse_result = parser.parse_file(save_path)

                if not parse_result.get("success", False):
                    failed_files.append((filename, parse_result.get("error", "Unknown error")))
                    continue

                # Chunk document
                doc_text = parse_result.get("text", "")
                metadata = parse_result.get("metadata", {})
                chunks = chunker.chunk_document(doc_text, metadata)

                # Incremental add to persistent ChromaDB (appends, does not overwrite)
                added_count = vector_store.add_documents(chunks)
                total_added_chunks += added_count
                success_files.append((filename, added_count))

                progress_bar.progress((idx + 1) / total_files)

            status_text.empty()
            progress_bar.progress(1.0)

            st.success(f"총 {len(success_files)}개 문서에서 {total_added_chunks}개 청크가 벡터 DB에 성공적으로 저장되었습니다!")

            if success_files:
                with st.expander("성공적으로 저장된 문서 목록", expanded=True):
                    for fname, count in success_files:
                        st.markdown(f"- **{fname}**: {count}개 청크 생성 완료")

            if failed_files:
                with st.expander("⚠️ 처리에 실패한 문서 목록", expanded=True):
                    for fname, err in failed_files:
                        st.error(f"- **{fname}**: {err}")

    # ------------------ TAB 2: DB STATS & MANAGEMENT ------------------
    with tabs[1]:
        st.subheader("📊 영구 벡터 데이터베이스 현황")
        stats = vector_store.get_stats()

        col_st1, col_st2 = st.columns(2)
        col_st1.metric("총 임베딩 청크 수", f"{stats['total_chunks']:,} 개")
        col_st2.metric("색인된 고유 문서 수", f"{stats['unique_documents_count']:,} 개")

        st.divider()
        st.subheader("📁 색인된 문서 관리")

        if stats["unique_documents_count"] == 0:
            st.info("현재 벡터 DB에 저장된 문서가 없습니다. 먼저 문서를 업로드해주세요.")
        else:
            doc_counts = stats.get("source_chunk_counts", {})
            for doc_name, count in doc_counts.items():
                col_d1, col_d2 = st.columns([8, 2])
                with col_d1:
                    st.markdown(f"📄 **{doc_name}** ({count}개 청크)")
                with col_d2:
                    if st.button("삭제", key=f"del_{doc_name}", type="secondary"):
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

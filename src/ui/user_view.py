"""
User Conversational RAG UI View for Streamlit.
Features interactive multi-turn dialogue, real-time streaming answer generation,
and source attribution/citations expander.
"""
from typing import List, Dict, Any
import streamlit as st
from src.rag.engine import ConversationalRAGEngine
from src.rag.vector_store import PersistentVectorStore
from src.security.logger import get_secure_logger

logger = get_secure_logger("user_view")


def render_user_view(rag_engine: ConversationalRAGEngine, vector_store: PersistentVectorStore):
    """Renders the conversational chat interface for end users."""
    st.header("💬 지식 기반 대화형 어시스턴트 (RAG Chat)")
    st.caption("업로드된 문서를 바탕으로 질문에 답변하며, 이전 대화 맥락을 기억합니다.")

    # 1. Sidebar configuration for search tuning
    with st.sidebar:
        st.subheader("⚙️ 검색 및 모델 설정")
        top_k = st.slider("참고할 청크 개수 (Top-K)", min_value=1, max_value=10, value=4, step=1)
        score_threshold = st.slider(
            "최소 유사도 기준 (Threshold)", min_value=0.0, max_value=0.9, value=0.1, step=0.05,
            help="유사도가 이 값 이상인 청크만 컨텍스트로 전달합니다."
        )

        st.divider()
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Display available indexed documents
        st.divider()
        st.subheader("📚 현재 등록된 문서 목록")
        stats = vector_store.get_stats()
        sources = stats.get("document_sources", [])
        if sources:
            for s in sources:
                st.markdown(f"- 📄 `{s}`")
        else:
            st.info("등록된 문서가 없습니다. 상단 관리자 탭에서 먼저 문서를 등록해주세요.")

    # 2. Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요! 업로드된 문서를 바탕으로 질문에 정확하게 답변해 드립니다. 궁금한 내용을 물어보세요.",
                "sources": []
            }
        ]

    # 3. Render prior conversation turns
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 참고 출처 및 컨텍스트 (Sources & Citations)", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"**[{idx+1}] {src['metadata'].get('source', '문서')}** "
                            f"(유사도: `{src.get('similarity', 0.0):.2f}`)"
                        )
                        st.text(src["text"])

    # 4. Handle incoming user query
    if user_query := st.chat_input("문서에 대해 질문하세요... (예: 생물작용제 및 독소 신고 제도의 제출처를 알려줘)"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Prepare assistant response container with visual indicator
        with st.chat_message("assistant"):
            status_box = st.status("💡 답변 준비 중...", expanded=True)
            response_placeholder = st.empty()
            sources_container = st.container()

            # Conversational history passed to RAG engine
            history_for_engine = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            full_response = ""
            retrieved_sources: List[Dict[str, Any]] = []

            # Stream generation
            generator = rag_engine.generate_streaming_response(
                query=user_query,
                chat_history=history_for_engine,
                top_k=top_k,
                score_threshold=score_threshold
            )

            for event in generator:
                event_type = event.get("type")
                if event_type == "status":
                    stage = event.get("stage")
                    msg = event.get("message")
                    if stage in ("searching", "generating"):
                        status_box.update(label=f"⏳ {msg}", state="running")
                    elif stage == "streaming":
                        status_box.update(label="✍️ 실시간 답변 출력 중...", state="running")
                    elif stage == "done":
                        status_box.update(label="✅ 답변 생성 완료", state="complete", expanded=False)
                elif event_type == "sources":
                    retrieved_sources = event.get("content", [])
                    if retrieved_sources:
                        status_box.write(f"📖 **{len(retrieved_sources)}개**의 관련 문서 청크를 찾았습니다.")
                elif event_type == "token":
                    full_response += event.get("content", "")
                    response_placeholder.markdown(full_response + " ▌")
                elif event_type == "error":
                    full_response += f"\n\n⚠️ {event.get('content')}"
                    response_placeholder.markdown(full_response)
                    status_box.update(label="⚠️ 오류 발생", state="error", expanded=True)

            # Finalize assistant text
            response_placeholder.markdown(full_response)

            # Render sources expander if available
            if retrieved_sources:
                with sources_container.expander("🔍 참고 출처 및 컨텍스트 (Sources & Citations)", expanded=False):
                    for idx, src in enumerate(retrieved_sources):
                        st.markdown(
                            f"**[{idx+1}] {src['metadata'].get('source', '문서')}** "
                            f"(유사도: `{src.get('similarity', 0.0):.2f}`)"
                        )
                        st.text(src["text"])

            # Store to session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_sources
            })

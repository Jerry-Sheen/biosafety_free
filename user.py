import os
"""
==============================================================================
Biosafety RAG System - User Interface & Unified CLI Entrypoint (user.py)
------------------------------------------------------------------------------
Provides dual-mode operation:
1. Powerful CLI for document ingestion, deduplication, OCR check, and Gemini inspection:
   - python user.py --input data/input/document.pdf
   - python user.py --input data/input --recursive
   - python user.py --input data/input/document.pdf --force
   - python user.py --check-ocr
   - python user.py --check-gemini
   - python user.py --list-models
2. Interactive Streamlit Conversational Web UI:
   - streamlit run user.py
==============================================================================
"""
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.security.secrets_manager import sanitize_api_key


def filter_pathogen_keys(query: str, keys: List[str]) -> List[str]:
    """
    입력된 글자(query)가 포함된 병원균 이름(Key 값)만 필터링합니다.
    대소문자를 구분하지 않으며(case-insensitive), 입력값의 따옴표(' 또는 ")를 자동 정제합니다.
    접두어 일치(prefix match) 항목을 우선 정렬하여 반환합니다.
    """
    if not query:
        return []
    q = query.strip().strip("'\"").strip().lower()
    if not q or not keys:
        return []
    matches = [k for k in keys if q in k.lower()]
    prefix_matches = sorted([k for k in matches if k.lower().startswith(q)], key=lambda x: x.lower())
    other_matches = sorted([k for k in matches if not k.lower().startswith(q)], key=lambda x: x.lower())
    return prefix_matches + other_matches


def get_pathogen_keys_from_vector_store(source_name: str = "병원균(위험균) 종류 v3.json") -> List[str]:
    """
    영구 벡터 DB(ChromaDB)에 저장된 특정 JSON 문서 청크들로부터
    병원균 이름(Key 값) 목록을 추출하여 반환합니다.
    ChromaDB 클라이언트를 직접 사용하여 즉각 로드합니다.
    """
    import re
    import json
    import chromadb

    # 1. 로컬 JSON 파일 직접 로드 (ChromaDB 락 충돌 방지 및 즉각 로드)
    json_path = PROJECT_ROOT / "data" / "uploads" / source_name
    if json_path.is_file():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return list(data.keys())
        except Exception:
            pass

    # 2. PersistentVectorStore 객체 경유
    try:
        from src.rag.vector_store import PersistentVectorStore
        vs = PersistentVectorStore()
        keys = vs.get_pathogen_keys(source_name)
        if keys:
            return keys
    except Exception:
        pass

    # 3. ChromaDB 직접 연결로 로드
    try:
        def _check_db(p: Path) -> bool:
            try:
                return (p / "chroma.sqlite3").is_file()
            except (OSError, PermissionError):
                return False

        target_dir = Path(config.chroma_persist_dir)
        if not _check_db(target_dir):
            for cand in [Path("data/chroma_db").resolve(), PROJECT_ROOT / "data" / "chroma_db", Path("/mount/src/biosafety-rag/data/chroma_db")]:
                if _check_db(cand):
                    target_dir = cand
                    break
        col = client.get_collection(config.chroma_collection_name)
        res = col.get(where={"source": source_name}, include=["documents"])
        docs = res.get("documents", [])
        pattern = re.compile(r'\[병원균/항목:\s*(.*?)(?=\]\s*\r?\n\s*-)', re.DOTALL)
        keys = []
        for d in docs:
            for k in pattern.findall(d):
                kc = k.strip()
                if kc and kc not in keys:
                    keys.append(kc)
        if keys:
            return keys
    except Exception:
        pass

    return []


def run_check_ocr():
    """Validates OCR installation and environment configuration."""
    from parsers.ocr_parser import (
        HAS_RAPIDOCR,
        HAS_PYTESSERACT,
        get_tesseract_executable
    )
    print("=" * 70)
    print("🔍 [1/2] RapidOCR Engine Check")
    print("=" * 70)
    if HAS_RAPIDOCR:
        print("✅ RapidOCR (ONNX Runtime): INSTALLED & READY")
        try:
            from parsers.ocr_parser import _get_rapidocr_instance
            engine = _get_rapidocr_instance()
            if engine:
                print("   - Engine initialized successfully with Korean/Multilingual model.")
            else:
                print("   ⚠️ Engine loaded but instance returned None.")
        except Exception as e:
            print(f"   ⚠️ RapidOCR check warning: {e}")
    else:
        print("❌ RapidOCR: NOT INSTALLED")
        print("   설치 명령: pip install rapidocr onnxruntime")

    print("\n" + "=" * 70)
    print("🔍 [2/2] Tesseract OCR Engine Check")
    print("=" * 70)
    tess_path = get_tesseract_executable()
    if HAS_PYTESSERACT and tess_path:
        print("✅ Tesseract OCR: INSTALLED & FOUND")
        print(f"   - Executable: {tess_path}")
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tess_path
            langs = pytesseract.get_languages()
            print(f"   - Installed Languages: {', '.join(langs)}")
            if "kor" in langs:
                print("   ✅ Korean language data (kor.traineddata): DETECTED")
            else:
                print("   ⚠️ Korean language data (kor.traineddata) is MISSING!")
                print("      Tessdata 다운로드: https://github.com/tesseract-ocr/tessdata")
        except Exception as e:
            print(f"   ⚠️ Could not query Tesseract languages: {e}")
    else:
        print("❌ Tesseract OCR: NOT FOUND OR NOT CONFIGURED")
        if not HAS_PYTESSERACT:
            print("   - pytesseract package is missing: pip install pytesseract")
        print("\n   [Windows 설치 안내]:")
        print("   1. Chocolatey 사용 시:")
        print("      choco install tesseract")
        print("   2. 수동 설치:")
        print("      UB-Mannheim 인스톨러 다운로드 (한국어 언어팩 체크):")
        print("      https://github.com/UB-Mannheim/tesseract/wiki")
        print("   3. .env 파일에 실행 경로 지정 (선택):")
        print(r"      TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe")
    print("=" * 70)


def run_check_gemini():
    """Validates Google Gemini API key and model connectivity."""
    from src.rag.gemini_client import GeminiModelResolver, classify_gemini_error
    print("=" * 70)
    print("🔑 Google Gemini API Connectivity Check")
    print("=" * 70)
    api_key = config.gemini_api_key
    if not api_key:
        print("❌ GEMINI_API_KEY is NOT CONFIGURED.")
        print("   .env 파일에 GEMINI_API_KEY=your_key_here 를 설정하거나 환경변수를 등록하세요.")
        print("   무료 키 발급: https://aistudio.google.com/app/apikey")
        return

    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "********"
    print(f"   - API Key Configured: {masked_key}")
    print(f"   - Configured Target Model: {config.gemini_model}")

    resolver = GeminiModelResolver(api_key=api_key)
    try:
        resolved, warn = resolver.resolve_model(config.gemini_model)
        if warn:
            print(f"   ⚠️ Model Resolution Notice: {warn}")
        print(f"   ✅ Active Target Model: {resolved}")

        # Quick test generation call
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=resolved,
            contents="Say 'OK' if you receive this message."
        )
        print("   ✅ Live API Call Success! Model Response: " + (resp.text or "").strip()[:50])
    except Exception as e:
        diag = classify_gemini_error(e)
        print(f"   ❌ Connection Check Failed! Error Type: {diag['error_type']}")
        print(f"   {diag['user_message']}")
        print(f"   Detailed: {str(e)}")
    print("=" * 70)


def run_list_models():
    """Lists available Gemini models and shows support for generateContent."""
    from src.rag.gemini_client import GeminiModelResolver
    print("=" * 70)
    print("📋 Google Gemini Available Models List")
    print("=" * 70)
    api_key = config.gemini_api_key
    if not api_key:
        print("❌ GEMINI_API_KEY is required to list available models.")
        print("   .env 파일에 GEMINI_API_KEY를 등록해주세요.")
        return

    resolver = GeminiModelResolver(api_key=api_key)
    models = resolver.list_available_models()

    if not models:
        print("⚠️ Could not retrieve model list or no models returned.")
        return

    header = f"{'Model Name':<35} | {'generateContent':<16} | {'Display Name'}"
    print(header)
    print("-" * len(header))
    for m in models:
        gen_status = "✅ YES" if m["supports_generate_content"] else "❌ NO"
        print(f"{m['short_name']:<35} | {gen_status:<16} | {m['display_name']}")
    print("=" * 70)


def run_ingestion(input_path_str: str, recursive: bool = False, force: bool = False):
    """Executes ingestion pipeline via CLI."""
    from pipeline.deduplication import ingest_document, ingest_directory
    from vectorstore.store import VectorStore
    input_path = Path(input_path_str)
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        sys.exit(1)

    vstore = VectorStore()
    print("=" * 70)
    print(f"🚀 Starting Ingestion Pipeline: {input_path}")
    print(f"   - Force overwrite: {force}")
    print(f"   - Recursive scan: {recursive}")
    print(f"   - Target ChromaDB: {config.chroma_persist_dir}")
    print("=" * 70)

    if input_path.is_file():
        res = ingest_document(input_path, force=force, vector_store=vstore)
        results = [res]
    else:
        results = ingest_directory(input_path, recursive=recursive, force=force, vector_store=vstore)

    print("\n" + "=" * 70)
    print("📊 Ingestion Summary Report")
    print("=" * 70)
    total_success = 0
    total_skipped = 0
    total_failed = 0
    total_chunks = 0

    for r in results:
        status_symbol = "✅" if r["status"] == "success" else ("⏭️" if r["status"] == "skipped" else "❌")
        print(
            f"{status_symbol} [{r['status'].upper():<7}] {r['source']} | "
            f"Pages: {r['pages']} (OCR: {r['ocr_pages']}) | Chunks: {r['chunks']}"
        )
        if r.get("warnings"):
            for w in r["warnings"]:
                print(f"   ⚠️ Warning: {w}")
        if r["status"] == "success":
            total_success += 1
            total_chunks += r["chunks"]
        elif r["status"] == "skipped":
            total_skipped += 1
        else:
            total_failed += 1

    print("-" * 70)
    print(f"Total Processed: {len(results)} | Success: {total_success} | Skipped: {total_skipped} | Failed: {total_failed}")
    print(f"Total New Chunks Stored: {total_chunks}")
    stats = vstore.get_stats()
    print(f"Total Collection Chunks Now: {stats['total_chunks']} ({stats['unique_documents_count']} documents)")
    print("=" * 70)


def run_streamlit_app():
    """Runs Streamlit Conversational Web UI."""
    import streamlit as st
    import traceback
    # Compatibility fallbacks for newer Streamlit APIs
    if not hasattr(st, "chat_message"):
        class _DummyChat:
            def __enter__(self):
                return st.container()
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
        st.chat_message = lambda role: _DummyChat()
    if not hasattr(st, "status"):
        class _DummyStatus:
            def __init__(self, label="", expanded=False):
                self.label = label
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
            def update(self, label=None, state=None, expanded=False):
                if label is not None:
                    st.write(label)
        st.status = lambda label="", expanded=False: _DummyStatus(label=label)



    # Set page config before any other Streamlit calls (safe if already set by app.py)
    try:
        st.set_page_config(
            page_title="SafeBio Navigator",
            page_icon="💬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass
    # Import core modules with error handling
    try:
        from src.rag.vector_store import PersistentVectorStore
        from src.rag.engine import ConversationalRAGEngine, inject_practical_guide_notice
    except Exception as e:
        st.error('❌ 필수 모듈 로드 실패')
        st.error(traceback.format_exc())
        st.stop()
    # Initialize sessions store directory
    sessions_dir = PROJECT_ROOT / "data" / "chat_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    def _save_session(session_id: str, messages: list):
        if not session_id or not messages:
            return
        # Only save if there is at least one actual user message (prevents empty ghost sessions)
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return
        filepath = sessions_dir / f"{session_id}.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _delete_session(session_id: str):
        filepath = sessions_dir / f"{session_id}.json"
        if filepath.is_file():
            try:
                filepath.unlink(missing_ok=True)
            except Exception:
                pass

    def _list_sessions():
        import datetime
        files = list(sessions_dir.glob("*.json"))
        # Sort by file modification time desc
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        session_list = []
        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    title = "새로운 대화"
                    for m in data:
                        if m.get("role") == "user":
                            title = m.get("content", "").strip()[:24]
                            break
                    mtime_dt = datetime.datetime.fromtimestamp(file.stat().st_mtime)
                    time_label = mtime_dt.strftime("%m/%d %H:%M")
                    session_list.append({
                        "id": file.stem,
                        "title": title,
                        "time_label": time_label,
                        "file": file,
                        "messages": data
                    })
            except Exception:
                pass
        return session_list

    # Default greeting message helper
    def _default_messages():
        return [
            {
                "role": "assistant",
                "content": "안녕하세요! KRIBB 생물안전 관리 AI 에이전트입니다. 병원균, 위험균, 생물작용제 및 독소 관련 법/제도, 신고 절차, 주의 사항 등에 관해 궁금한 점을 질문해 주세요."
            }
        ]

    # Initialize current session if not set
    if "current_session_id" not in st.session_state or "user_messages" not in st.session_state:
        existing = _list_sessions()
        if existing:
            # Load the most recent session directly so continuing conversation stays in this session!
            st.session_state.current_session_id = existing[0]["id"]
            st.session_state.user_messages = existing[0]["messages"]
        else:
            import datetime
            st.session_state.current_session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.user_messages = _default_messages()

    # Fast pathogen keys loader cached without blocking on embedding model
    @st.cache_data(show_spinner=False)
    def _load_cached_pathogen_keys() -> List[str]:
        return get_pathogen_keys_from_vector_store("병원균(위험균) 종류 v3.json")

    # Initialize vector store with st.cache_resource to prevent OOM crash
    @st.cache_resource(show_spinner="지식 베이스(ChromaDB)를 연결하는 중...")
    def _get_vector_store():
        return PersistentVectorStore()

    try:
        vector_store = _get_vector_store()
        rag_engine = ConversationalRAGEngine(
            vector_store=vector_store,
            llm_provider="free",
            gemini_api_key="",
            openai_api_key=""
        )
    except Exception as e:
        st.error('❌ 벡터스토어 또는 엔진 초기화 실패')
        st.error(traceback.format_exc())
        st.stop()

    # ------------------ [SIDEBAR : LEFT PANEL] ------------------
    with st.sidebar:
        st.markdown("### 💬 대화 관리")
        
        # '새채팅' 버튼
        if st.button("➕ 새채팅 시작", type="primary", use_container_width=True):
            # Save current session before reset if it has conversation
            _save_session(st.session_state.current_session_id, st.session_state.user_messages)
            import datetime
            st.session_state.current_session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.user_messages = _default_messages()
            st.rerun()

        st.divider()

        # LLM 모델 및 옵션 설정
        st.markdown("### ⚙️ 서비스 모드 및 AI 설정")
        service_mode = st.radio(
            "선택할 서비스 모드:",
            options=[
                "🆓 완전 무료 모드 (로컬 오픈소스 AI)",
                "💳 유료 고성능 모드 (개인 API 키 사용)"
            ],
            index=0,
            help="무료 모드는 API 비용 없이 이용 가능하며, 유료 모드는 본인의 OpenAI/Gemini 키로 고성능 추론을 수행합니다."
        )

        if "user_gemini_api_key" not in st.session_state:
            st.session_state.user_gemini_api_key = ""
        if "user_openai_api_key" not in st.session_state:
            st.session_state.user_openai_api_key = ""

        if "무료" in service_mode:
            rag_engine.llm_provider = "free"
            st.success("✅ **외부 사용자 완전 무료 모드**가 활성화되었습니다. 별도 API 키 없이 즉시 무료로 질문하실 수 있습니다.")
            st.caption(f"🎯 로컬 CPU 모델: `{getattr(config, 'local_llm_model', 'Qwen/Qwen2.5-1.5B-Instruct')}`")
        else:
            provider_option = st.radio(
                "사용할 유료 LLM 제공자:",
                options=[
                    "✨ Google Gemini",
                    "🟢 OpenAI (ChatGPT)"
                ],
                index=0
            )
            if "Gemini" in provider_option:
                rag_engine.llm_provider = "gemini"
                api_key_input = st.text_input(
                    "🔑 Gemini API 키 입력:",
                    value=st.session_state.user_gemini_api_key,
                    type="password",
                    placeholder="AIzaSy...",
                    help="Google AI Studio에서 무료로 발급받은 API 키를 입력해 주세요. (개인 키는 서버에 저장되지 않습니다)"
                )
                raw_gemini = api_key_input.strip()
                cleaned_gemini = sanitize_api_key(raw_gemini)
                if raw_gemini and raw_gemini != cleaned_gemini:
                    st.info("💡 입력된 Gemini API 키에서 불필요한 따옴표/공백/접두사를 자동으로 정제했습니다.")
                st.session_state.user_gemini_api_key = cleaned_gemini
                rag_engine.gemini_api_key = cleaned_gemini or sanitize_api_key(config.gemini_api_key) or ""
                rag_engine.gemini_model_resolver.api_key = rag_engine.gemini_api_key

                if not rag_engine.gemini_api_key:
                    st.info("💡 질문을 하려면 본인의 Gemini API 키가 필요합니다.\n\n👉 [Google AI Studio에서 무료 키 발급받기](https://aistudio.google.com/app/apikey)")
                st.caption(f"🎯 모델: `{rag_engine.gemini_model}`")
            else:
                rag_engine.llm_provider = "openai"
                api_key_input = st.text_input(
                    "🔑 OpenAI API 키 입력:",
                    value=st.session_state.user_openai_api_key,
                    type="password",
                    placeholder="sk-...",
                    help="OpenAI 플랫폼에서 발급받은 API 키를 입력해 주세요. (개인 키는 서버에 저장되지 않습니다)"
                )
                raw_openai = api_key_input.strip()
                cleaned_openai = sanitize_api_key(raw_openai)
                if raw_openai and raw_openai != cleaned_openai:
                    st.info("💡 입력된 OpenAI API 키에서 불필요한 따옴표/공백/유니코드 대시를 자동으로 정제했습니다.")
                st.session_state.user_openai_api_key = cleaned_openai
                active_key = cleaned_openai or sanitize_api_key(config.openai_api_key) or ""
                rag_engine.openai_api_key = active_key
    
                # OpenAI Model selection (gpt-4o-mini supports 128k context to avoid context_length_exceeded)
                if "user_openai_model" not in st.session_state:
                    st.session_state.user_openai_model = getattr(config, "openai_model", "gpt-4o-mini")
    
                openai_models = [
                    "gpt-4o-mini",
                    "gpt-4o",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo"
                ]
                if st.session_state.user_openai_model not in openai_models:
                    openai_models.insert(0, st.session_state.user_openai_model)
    
                selected_model = st.selectbox(
                    "🎯 OpenAI 모델 선택:",
                    options=openai_models,
                    index=openai_models.index(st.session_state.user_openai_model),
                    help="gpt-4o-mini는 128k 토큰 컨텍스트를 지원하여 대용량 법령 및 지침서 데이터를 오류 없이 안정적으로 처리합니다. (기본 추천)"
                )
                st.session_state.user_openai_model = selected_model
                rag_engine.openai_model = selected_model
    
                if active_key:
                    if not active_key.isascii():
                        st.error("⚠️ API 키에 한글 또는 유효하지 않은 문자가 포함되어 있습니다. 올바른 영문 API 키를 입력해 주세요.")
                    from src.rag.openai_client import OpenAIClient
                    if rag_engine.openai_client:
                        rag_engine.openai_client.set_api_key(active_key)
                        rag_engine.openai_client.model = rag_engine.openai_model
                    else:
                        rag_engine.openai_client = OpenAIClient(api_key=active_key, model=rag_engine.openai_model)
                else:
                    rag_engine.openai_client = None
                    st.info("💡 질문을 하려면 본인의 OpenAI API 키가 필요합니다.\n\n👉 [OpenAI 플랫폼에서 키 발급받기](https://platform.openai.com/api-keys)")
    
                token_desc = "128k 토큰 지원 (추천)" if "4" in rag_engine.openai_model else "16k 토큰 (긴 법령 검색 시 초과 주의)"
                st.caption(f"🟢 모델: `{rag_engine.openai_model}` ({token_desc})")
    
        top_k = st.slider("검색 청크 수 (Top‑K)", min_value=1, max_value=10, value=5, step=1)

        st.divider()
        stats = vector_store.get_stats()
        # Self-healing: if cached store reports 0 chunks, clear cache and reconnect immediately
        if stats.get('total_chunks', 0) == 0:
            st.cache_resource.clear()
            from src.rag.vector_store import PersistentVectorStore
            vector_store = PersistentVectorStore()
            stats = vector_store.get_stats()

        st.markdown("### 📊 지식 베이스 현황")
        st.metric("총 색인 청크 수", f"{stats['total_chunks']:,} 개")
        st.metric("저장된 문서 수", f"{stats['unique_documents_count']} 권")
        if stats['total_chunks'] == 0:
            st.warning("⚠️ 지식 베이스 청크가 0개입니다.")
            if st.button("🔄 지식 베이스 다시 연결", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()

        st.divider()

        # 사이드바 하단: 저장된 이전 대화 목록
        st.markdown("### 📂 저장된 이전 대화")
        saved_sessions = _list_sessions()
        if not saved_sessions:
            st.caption("저장된 이전 대화가 없습니다.")
        else:
            for s in saved_sessions:
                col_btn, col_del = st.columns([8, 2])
                with col_btn:
                    is_current = (s["id"] == st.session_state.current_session_id)
                    time_suffix = f" ({s['time_label']})" if s.get("time_label") else ""
                    btn_label = f"{'👉 ' if is_current else '📝 '}{s['title']}{time_suffix}"
                    if st.button(btn_label, key=f"sess_{s['id']}", use_container_width=True):
                        if not is_current:
                            # Save current session before switching if it has user messages
                            _save_session(st.session_state.current_session_id, st.session_state.user_messages)
                            st.session_state.current_session_id = s["id"]
                            try:
                                with open(s["file"], "r", encoding="utf-8") as sf:
                                    st.session_state.user_messages = json.load(sf)
                            except Exception:
                                st.session_state.user_messages = s["messages"]
                            st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{s['id']}", help="대화 삭제"):
                        _delete_session(s["id"])
                        if s["id"] == st.session_state.current_session_id:
                            remaining = [item for item in _list_sessions() if item["id"] != s["id"]]
                            if remaining:
                                st.session_state.current_session_id = remaining[0]["id"]
                                st.session_state.user_messages = remaining[0]["messages"]
                            else:
                                import datetime
                                st.session_state.current_session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                st.session_state.user_messages = _default_messages()
                        st.toast("대화가 완전히 삭제되었습니다.", icon="🗑️")
                        st.rerun()

    # ------------------ [MAIN WINDOW : RIGHT PANEL] ------------------
    st.markdown(
        """
        <style>
        /* 메인 컨테이너 상단 여백: 헤더가 상단에서 1칸 정도 알맞게 내려오도록 조정 */
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 7.0rem !important;
        }
        header[data-testid="stHeader"] {
            height: 1.5rem !important;
            min-height: 1.5rem !important;
            background: transparent !important;
        }
        .user-subheader { font-size: 25px; font-weight: bold; color: #4B5563; margin-top: 0.2rem; margin-bottom: 2px; padding-left: 90px; line-height: 1.2; }
        .user-header { font-weight: 700; color: #1E3A8A; margin-top: 2px; margin-bottom: 20px; line-height: 1.2; }
        .user-desc { color: #4B5563; font-size: 1.0rem; margin-bottom: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="user-subheader" style="font-size:25px; font-weight:bold; padding-left:90px; margin-bottom: 2px;">KRIBB Biosafety AI Agent</div>
        <div class="user-header" style="margin-top: 2px; margin-bottom: 20px;">
            <span style="font-size:45px; font-weight:bold; color:#1E3A8A;">🧭 SafeBio Navigator,</span>&nbsp;&nbsp;&nbsp;&nbsp;
            <span style="font-size:28px; font-weight:bold; color:#1E3A8A;">안전한 바이오 연구를 위한 규제 길잡이</span>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Display conversation messages in main container
    for msg in st.session_state.user_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 참고 출처 및 컨텍스트", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"**[{idx+1}] {src['metadata'].get('source', '문서')}** (유사도: `{src.get('similarity', 0.0):.2f}`)")
                        st.text(src["text"][:400] + "...")

    # Load pathogen keys for autocomplete
    pathogen_keys = sorted(_load_cached_pathogen_keys(), key=lambda x: x.lower())

    # HTML5 Datalist Native Autocomplete Component
    import streamlit.components.v1 as components
    import html as html_lib

    options_html = "\n".join([f'<option value="{html_lib.escape(k)}">' for k in pathogen_keys])
    keys_json = json.dumps(pathogen_keys, ensure_ascii=False)

    autocomplete_component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    </head>
    <body style="margin: 0; padding: 0; display: none;">
    <script>
    (function() {{
        const allKeys = {keys_json};

        // Safely create and maintain a standalone dock bar above stChatInput without touching React DOM nodes
        function initDockBar() {{
            try {{
                const parentDoc = window.parent.document;
                const chatInput = parentDoc.querySelector('[data-testid="stChatInput"]');
                if (!chatInput || !chatInput.parentElement) return;

                let dock = parentDoc.getElementById('st-pathogen-bottom-bar');
                const needsBuild = !dock || !dock.querySelector('#pathogen-dock-dropdown');

                if (needsBuild) {{
                    if (!dock) {{
                        dock = parentDoc.createElement('div');
                        dock.id = 'st-pathogen-bottom-bar';
                        chatInput.parentElement.insertBefore(dock, chatInput);
                    }}
                    dock.style.cssText = 'width: 100%; margin-bottom: 6px; padding: 0; box-sizing: border-box; position: relative;';
                    dock.innerHTML = `
                        <!-- Popup dropdown list sized to 2/3 of dock bar, popping upwards directly above dock -->
                        <div id="pathogen-dock-dropdown"
                             style="display: none; position: absolute; left: 0; bottom: 100%; margin-bottom: 6px; width: 66.7%; min-width: 320px; max-width: 100%; max-height: 320px; overflow-y: auto; background: #ffffff; border: 1.5px solid #3B82F6; border-radius: 8px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.22); z-index: 999999; font-size: 13px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                        </div>
                        <div style="background: #F8FAFC; border: 1.5px solid #CBD5E1; border-radius: 9px; padding: 6px 12px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 4px; margin-bottom: 4px;">
                                <span style="font-size: 13px; font-weight: 700; color: #1E3A8A; display: flex; align-items: center; gap: 6px;">
                                    🔍 병원균/바이러스 명칭 자동완성 검색:
                                </span>
                                <span style="font-size: 11px; color: #64748B;">
                                    클릭 또는 타이핑 시 병원균 목록이 표시되며, 선택 시 채팅창에 반영됩니다.
                                </span>
                            </div>
                            <div style="display: flex; gap: 8px; align-items: center;">
                                <input type="text" id="pathogen-dock-input"
                                       placeholder="병원균 이름 입력 (예: ab, abr, a, e, 아브린)..." autocomplete="off"
                                       style="flex: 1; min-width: 0; padding: 6px 12px; font-size: 13px; border: 1.5px solid #3B82F6; border-radius: 6px; outline: none; background: #FFFFFF; color: #0F172A;" />
                                <button id="btn-dock-apply"
                                        style="background-color: #2563EB; color: #FFFFFF; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap; transition: background-color 0.15s; flex-shrink: 0;">
                                    채팅창에 적용
                                </button>
                            </div>
                        </div>
                    `;
                }} else {{
                    if (chatInput.parentElement && chatInput.previousElementSibling !== dock) {{
                        chatInput.parentElement.insertBefore(dock, chatInput);
                    }}
                }}

                const dockInput = dock.querySelector('#pathogen-dock-input');
                const dockBtn = dock.querySelector('#btn-dock-apply');
                const dockDropdown = dock.querySelector('#pathogen-dock-dropdown');

                if (!dockInput || !dockBtn || !dockDropdown) return;

                // Prevent periodic timer from resetting active selection while dropdown is in use
                if (dockInput._boundWindow === window && dockDropdown.style.display !== 'none') {{
                    return;
                }}
                dockInput._boundWindow = window;

                let selectedIndex = parseInt(dockDropdown.dataset.selectedIndex || '-1', 10);

                function getDropdownItems() {{
                    return dockDropdown ? dockDropdown.querySelectorAll('.dock-dropdown-item') : [];
                }}

                function updateSelection(newIdx, scrollIntoView = true, syncInput = true) {{
                    const items = getDropdownItems();
                    if (!items.length) {{
                        selectedIndex = -1;
                        if (dockDropdown) dockDropdown.dataset.selectedIndex = '-1';
                        return;
                    }}

                    items.forEach((el, idx) => {{
                        if (idx === newIdx) {{
                            el.classList.add('dock-selected');
                            el.style.backgroundColor = '#DBEAFE';
                            el.style.color = '#1D4ED8';
                            el.style.fontWeight = '600';
                        }} else {{
                            el.classList.remove('dock-selected');
                            el.style.backgroundColor = '#FFFFFF';
                            el.style.color = '#1E293B';
                            el.style.fontWeight = 'normal';
                        }}
                    }});

                    selectedIndex = newIdx;
                    if (dockDropdown) dockDropdown.dataset.selectedIndex = String(newIdx);

                    if (selectedIndex >= 0 && selectedIndex < items.length) {{
                        const activeEl = items[selectedIndex];
                        if (scrollIntoView) {{
                            activeEl.scrollIntoView({{ block: 'nearest' }});
                        }}
                        if (syncInput && activeEl.textContent) {{
                            dockInput.value = activeEl.textContent.trim();
                        }}
                    }}
                }}

                function applyValue(val) {{
                    if (!val) return;
                    const cleanVal = val.trim();
                    if (!cleanVal) return;
                    const ta = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                    if (ta) {{
                        const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value')?.set ||
                                       Object.getOwnPropertyDescriptor(ta, 'value')?.set;
                        if (setter) {{
                            setter.call(ta, cleanVal);
                        }} else {{
                            ta.value = cleanVal;
                        }}
                        ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        ta.focus();
                    }}
                }}

                function renderDropdown(filterVal = '') {{
                    selectedIndex = -1;
                    if (dockDropdown) dockDropdown.dataset.selectedIndex = '-1';
                    const q = (filterVal || '').trim().replace(/^['"]+|['"]+$/g, '').toLowerCase();
                    let matches;
                    if (!q) {{
                        matches = allKeys;
                    }} else {{
                        const fullMatches = allKeys.filter(k => k.toLowerCase().includes(q));
                        const prefix = fullMatches.filter(k => k.toLowerCase().startsWith(q));
                        const others = fullMatches.filter(k => !k.toLowerCase().startsWith(q));
                        matches = prefix.concat(others);
                    }}

                    if (matches.length === 0) {{
                        dockDropdown.innerHTML = '<div style="padding: 10px 14px; color: #94A3B8; font-size: 12px;">일치하는 병원균 명칭이 없습니다.</div>';
                    }} else {{
                        dockDropdown.innerHTML = '';
                        const headerRow = parentDoc.createElement('div');
                        headerRow.style.cssText = 'padding: 6px 12px; background: #F1F5F9; color: #475569; font-size: 11px; font-weight: 600; border-bottom: 1px solid #E2E8F0; position: sticky; top: 0; z-index: 10; display: flex; justify-content: space-between; align-items: center;';
                        headerRow.innerHTML = q
                            ? `<span>🔍 검색 결과</span><span style="color: #2563EB;">총 ${{matches.length}}개 (방향키 ↑↓ / 엔터 / 클릭 선택)</span>`
                            : `<span>📋 전체 병원균/바이러스 목록</span><span style="color: #2563EB;">총 ${{matches.length}}개 (방향키 ↑↓ / 엔터 / 클릭 선택)</span>`;
                        dockDropdown.appendChild(headerRow);

                        matches.forEach((item, idx) => {{
                            const row = parentDoc.createElement('div');
                            row.className = 'dock-dropdown-item';
                            row.dataset.index = idx;
                            row.style.cssText = 'padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #F1F5F9; color: #1E293B; line-height: 1.4; word-break: break-word; transition: background 0.1s;';
                            row.textContent = item;
                            row.onmouseover = () => {{
                                updateSelection(idx, false, false);
                            }};
                            row.onmousedown = (ev) => {{
                                ev.preventDefault();
                                dockInput.value = item;
                                applyValue(item);
                                dockDropdown.style.display = 'none';
                                dockDropdown.dataset.selectedIndex = '-1';
                                selectedIndex = -1;
                            }};
                            dockDropdown.appendChild(row);
                        }});
                    }}
                    dockDropdown.style.display = 'block';
                }}

                // Property assignment guarantees fresh active listeners on rerun without stale dead closures
                dockInput.onfocus = () => renderDropdown(dockInput.value);
                dockInput.onclick = () => renderDropdown(dockInput.value);
                dockInput.oninput = (e) => renderDropdown(e.target.value);

                dockInput.onkeydown = (e) => {{
                    const items = getDropdownItems();
                    const isOpen = dockDropdown && dockDropdown.style.display !== 'none';
                    const isEnter = e.key === 'Enter' || e.keyCode === 13 || e.which === 13 || e.code === 'Enter' || e.code === 'NumpadEnter';

                    if (e.key === 'ArrowDown') {{
                        e.preventDefault();
                        if (!isOpen) {{
                            renderDropdown(dockInput.value);
                            updateSelection(0, true, true);
                        }} else if (items.length > 0) {{
                            let next = selectedIndex + 1;
                            if (next >= items.length) next = 0;
                            updateSelection(next, true, true);
                        }}
                    }} else if (e.key === 'ArrowUp') {{
                        e.preventDefault();
                        if (!isOpen) {{
                            renderDropdown(dockInput.value);
                            updateSelection(items.length - 1, true, true);
                        }} else if (items.length > 0) {{
                            let prev = selectedIndex - 1;
                            if (prev < 0) prev = items.length - 1;
                            updateSelection(prev, true, true);
                        }}
                    }} else if (isEnter) {{
                        e.preventDefault();
                        e.stopPropagation();

                        let targetVal = '';
                        const selectedEl = dockDropdown ? dockDropdown.querySelector('.dock-selected') : null;

                        if (selectedEl && selectedEl.textContent) {{
                            targetVal = selectedEl.textContent.trim();
                        }} else if (selectedIndex >= 0 && selectedIndex < items.length) {{
                            targetVal = items[selectedIndex].textContent.trim();
                        }} else if (dockDropdown && dockDropdown.dataset.selectedIndex >= 0) {{
                            const savedIdx = parseInt(dockDropdown.dataset.selectedIndex, 10);
                            if (savedIdx >= 0 && savedIdx < items.length) {{
                                targetVal = items[savedIdx].textContent.trim();
                            }}
                        }}

                        if (!targetVal) {{
                            if (dockInput.value.trim()) {{
                                targetVal = dockInput.value.trim();
                            }} else if (isOpen && items.length > 0) {{
                                targetVal = items[0].textContent.trim();
                            }}
                        }}

                        if (targetVal) {{
                            dockInput.value = targetVal;
                            applyValue(targetVal);
                        }}

                        if (dockDropdown) {{
                            dockDropdown.style.display = 'none';
                            dockDropdown.dataset.selectedIndex = '-1';
                        }}
                        selectedIndex = -1;
                    }} else if (e.key === 'Escape') {{
                        if (dockDropdown) {{
                            dockDropdown.style.display = 'none';
                            dockDropdown.dataset.selectedIndex = '-1';
                        }}
                        selectedIndex = -1;
                    }}
                }};

                dockBtn.onclick = (e) => {{
                    e.preventDefault();
                    applyValue(dockInput.value);
                    if (dockDropdown) {{
                        dockDropdown.style.display = 'none';
                        dockDropdown.dataset.selectedIndex = '-1';
                    }}
                    selectedIndex = -1;
                }};
                dockBtn.onmouseover = () => {{ dockBtn.style.backgroundColor = '#1D4ED8'; }};
                dockBtn.onmouseout = () => {{ dockBtn.style.backgroundColor = '#2563EB'; }};

                if (!parentDoc._dockDismissAttached) {{
                    parentDoc._dockDismissAttached = true;
                    parentDoc.addEventListener('mousedown', (e) => {{
                        const d = parentDoc.getElementById('st-pathogen-bottom-bar');
                        const dd = parentDoc.getElementById('pathogen-dock-dropdown');
                        if (d && dd && !d.contains(e.target)) {{
                            dd.style.display = 'none';
                        }}
                    }});
                }}
            }} catch (err) {{
                console.warn("Pathogen dock attachment notice:", err);
            }}
        }}

        initDockBar();
        setTimeout(initDockBar, 50);
        setTimeout(initDockBar, 200);
        setTimeout(initDockBar, 600);
        setInterval(initDockBar, 1000);

        // Live autocomplete dropdown attached directly to parent stChatInputTextArea
        try {{
            const parentDoc = window.parent.document;
            let dropdown = parentDoc.getElementById('st-pathogen-dropdown');
            if (!dropdown) {{
                dropdown = parentDoc.createElement('div');
                dropdown.id = 'st-pathogen-dropdown';
                dropdown.style.cssText = 'position:fixed; display:none; max-height:220px; overflow-y:auto; background:#ffffff; border:1.5px solid #3B82F6; border-radius:8px; box-shadow:0 10px 25px rgba(0,0,0,0.18); z-index:999999; font-size:13px; font-family:sans-serif; width:calc(100% - 40px); max-width:800px;';
                parentDoc.body.appendChild(dropdown);
            }}

            const attachListener = () => {{
                const ta = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (!ta || ta._pathogenAttached) return;
                ta._pathogenAttached = true;

                ta.addEventListener('input', (e) => {{
                    const rawVal = e.target.value || '';
                    const q = rawVal.trim().replace(/^['"]+|['"]+$/g, '').toLowerCase();
                    if (!q) {{
                        dropdown.style.display = 'none';
                        return;
                    }}

                    const matches = allKeys.filter(k => k.toLowerCase().includes(q));
                    const prefix = matches.filter(k => k.toLowerCase().startsWith(q)).sort();
                    const others = matches.filter(k => !k.toLowerCase().startsWith(q)).sort();
                    const filtered = prefix.concat(others);

                    if (filtered.length === 0) {{
                        dropdown.style.display = 'none';
                        return;
                    }}

                    dropdown.innerHTML = '';
                    filtered.slice(0, 30).forEach(item => {{
                        const row = parentDoc.createElement('div');
                        row.style.cssText = 'padding:8px 12px; cursor:pointer; border-bottom:1px solid #F1F5F9; color:#1E293B;';
                        row.textContent = item;
                        row.addEventListener('mouseover', () => {{ row.style.background = '#EFF6FF'; row.style.color = '#1D4ED8'; }});
                        row.addEventListener('mouseout', () => {{ row.style.background = '#FFFFFF'; row.style.color = '#1E293B'; }});
                        row.addEventListener('mousedown', (ev) => {{
                            ev.preventDefault();
                            const ta = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                            if (ta) {{
                                const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value')?.set ||
                                               Object.getOwnPropertyDescriptor(ta, 'value')?.set;
                                if (setter) setter.call(ta, item); else ta.value = item;
                                ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                ta.focus();
                            }}
                            dropdown.style.display = 'none';
                        }});
                        dropdown.appendChild(row);
                    }});

                    const rect = ta.getBoundingClientRect();
                    dropdown.style.left = rect.left + 'px';
                    dropdown.style.width = Math.min(rect.width, 700) + 'px';
                    dropdown.style.bottom = (window.innerHeight - rect.top + 6) + 'px';
                    dropdown.style.display = 'block';
                }});

                ta.addEventListener('blur', () => {{
                    setTimeout(() => {{ dropdown.style.display = 'none'; }}, 250);
                }});
            }};

            attachListener();
            setInterval(attachListener, 1000);
        }} catch(err) {{
            console.error("Parent textarea dropdown attachment error:", err);
        }}
    }})();
    </script>
    </body>
    </html>
    """

    components.html(autocomplete_component_html, height=0)

    # Chat input box
    if prompt := st.chat_input("병원균, 위험균, 생물작용제 및 독소에 대한 질문을 입력하세요..."):
        # Check API key configuration before running RAG search
        # Check API key configuration only when paid cloud provider is selected
        if rag_engine.llm_provider in ("gemini", "openai"):
            current_api_key = rag_engine.gemini_api_key if rag_engine.llm_provider == "gemini" else rag_engine.openai_api_key
            if not current_api_key or not current_api_key.strip():
                st.session_state.user_messages.append({"role": "user", "content": prompt})
                provider_label = "Google Gemini" if rag_engine.llm_provider == "gemini" else "OpenAI"
                key_url = "https://aistudio.google.com/app/apikey" if rag_engine.llm_provider == "gemini" else "https://platform.openai.com/api-keys"
                st.session_state.user_messages.append({
                    "role": "assistant",
                    "content": (
                        f"⚠️ **{provider_label} API 키가 설정되지 않았습니다.**\n\n"
                        f"답변을 생성하기 위해서는 사용자의 API 키가 필요합니다.\n\n"
                        f"1. 좌측 사이드바의 **'🔑 {provider_label} API 키 입력'** 란에 본인의 API 키를 입력해 주세요.\n"
                        f"2. 아직 키가 없으시다면 [👉 {provider_label} API 키 발급받기]({key_url})에서 무료로 발급받으실 수 있습니다.\n\n"
                        f"*(※ 입력하신 API 키는 브라우저 세션 중에만 임시 사용되며, 서버에 일절 저장되지 않습니다.)*"
                    )
                })
                _save_session(st.session_state.current_session_id, st.session_state.user_messages)
                st.rerun()

        st.session_state.user_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_box = st.status("🔍 관련 문서 검색 중...", expanded=True)
            response_placeholder = st.empty()
            full_response = ""
            retrieved_sources = []
            has_error = False

            # FastAPI 백엔드 연동 및 자동 폴백 스트리밍 생성기
            def _get_streaming_events():
                import httpx
                import json
                backend_available = False
                backend_url = getattr(config, "backend_url", "http://localhost:8600")

                # Try FastAPI backend first
                try:
                    chat_req = {
                        "query": prompt,
                        "chat_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.user_messages[:-1]
                        ],
                        "mode": "free" if rag_engine.llm_provider in ("free", "local") else "paid",
                        "provider": rag_engine.llm_provider if rag_engine.llm_provider in ("gemini", "openai") else None,
                        "api_key": rag_engine.gemini_api_key if rag_engine.llm_provider == "gemini" else rag_engine.openai_api_key,
                        "model": rag_engine.gemini_model if rag_engine.llm_provider == "gemini" else rag_engine.openai_model
                    }
                    with httpx.Client(timeout=180.0) as client:
                        with client.stream("POST", f"{backend_url}/api/v1/chat", json=chat_req) as response:
                            if response.status_code == 200:
                                backend_available = True
                                for line in response.iter_lines():
                                    line_str = line.strip()
                                    if line_str.startswith("data: "):
                                        data_json = line_str[6:]
                                        try:
                                            yield json.loads(data_json)
                                        except Exception:
                                            pass
                except Exception:
                    backend_available = False

                # Fallback to direct in-process engine if backend is offline
                if not backend_available:
                    for ev in rag_engine.generate_streaming_response(
                        query=prompt,
                        chat_history=st.session_state.user_messages[:-1],
                        top_k=top_k,
                    ):
                        yield ev

            stream_gen = _get_streaming_events()
            for event in stream_gen:
                event_type = event.get("type")
                if event_type == "status":
                    status_box.update(label=f"⏳ {event.get('message')}", state="running")
                elif event_type == "sources":
                    retrieved_sources = event.get("content", [])
                elif event_type == "token":
                    token = event.get("content") or ""
                    full_response += token
                    response_placeholder.markdown(full_response + " ▌")
                elif event_type == "error":
                    err_msg = event.get('content', '')
                    status_box.update(label=f"⚠️ 오류 발생: {err_msg}", state="error", expanded=True)
                    full_response += f"\n\n{err_msg}"
                    response_placeholder.markdown(full_response)
                    has_error = True
                    break

            if not has_error and not full_response.strip():
                status_box.update(label="⚠️ 답변이 비어 있습니다. 선택한 모델에서 결과를 생성하지 못했습니다.", state="error", expanded=True)
                full_response = "⚠️ 답변이 비어 있습니다. 모델이 해당 질문에 대한 응답을 생성하지 못했을 수 있습니다. 질문을 다르게 표현하거나 문서가 충분히 관련되어 있는지 확인해 주세요."

            if not has_error:
                full_response = inject_practical_guide_notice(full_response)
                status_box.update(label="✅ 답변 생성 완료", state="complete", expanded=False)
            response_placeholder.markdown(full_response)

            if not has_error and retrieved_sources:
                with st.expander("🔍 참고 문서 청크", expanded=False):
                    for idx, src in enumerate(retrieved_sources):
                        st.markdown(f"**[{idx+1}] {src['metadata'].get('source', '문서')}** (유사도: `{src.get('similarity',0.0):.2f}`)")
                        st.text(src['text'][:400] + "...")

            st.session_state.user_messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_sources,
            })

            # Save to current session file
            _save_session(st.session_state.current_session_id, st.session_state.user_messages)
            





def main():
    parser = argparse.ArgumentParser(
        description="Biosafety RAG Pipeline CLI & Conversational Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python user.py --input data/input/document.pdf
  python user.py --input data/input --recursive
  python user.py --input data/input/document.pdf --force
  python user.py --check-ocr
  python user.py --check-gemini
  python user.py --list-models
  python user.py --ui (또는 streamlit run user.py)
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        help="수집 및 임베딩할 단일 문서 파일 경로 또는 디렉토리 경로"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="디렉토리 입력 시 하위 디렉토리까지 재귀적으로 탐색하여 문서를 수집합니다."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 벡터 DB에 존재하는 동일 문서(해시 일치)라도 강제로 이전 데이터를 삭제하고 다시 파싱·수집합니다."
    )
    parser.add_argument(
        "--check-ocr",
        action="store_true",
        help="RapidOCR 및 Tesseract OCR 엔진의 설치 상태, 실행 파일 경로, 언어 데이터(kor)를 점검합니다."
    )
    parser.add_argument(
        "--check-gemini",
        action="store_true",
        help="Google Gemini API 연결 상태, API 키 유효성, 추천 모델 통신 여부를 점검합니다."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Google Gemini API에서 현재 사용 가능한 모델 목록과 generateContent 지원 여부를 조회합니다."
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Streamlit 대화형 웹 UI를 실행합니다."
    )
    # OpenAI API 키를 직접 지정합니다.
    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API 키를 직접 지정합니다. .env에 설정되지 않은 경우 사용합니다."
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="영구 벡터 DB의 '병원균(위험균) 종류 v3.json'에서 특정 글자가 포함된 병원균 이름(Key 값)을 필터링하여 JSON 형식으로 출력합니다."
    )

# NOTE: The Streamlit UI is now launched explicitly via the CLI `--ui` flag or
# by running `streamlit run admin.py`. The automatic detection of a Streamlit
# runtime caused `user.py` to start its UI when imported (e.g., when admin.py
# imports the project package). This block is removed to prevent the user UI
# from appearing unexpectedly.
# The `main()` function below still supports the `--ui` argument for direct
# execution.
# is_streamlit_runtime = False  # Legacy auto‑run logic disabled

    args, _ = parser.parse_known_args()

    # Determine which mode to run based on provided arguments.
    if args.filter is not None:
        keys = get_pathogen_keys_from_vector_store("병원균(위험균) 종류 v3.json")
        filtered = filter_pathogen_keys(args.filter, keys)
        print(json.dumps(filtered, ensure_ascii=False))
    elif args.check_ocr:
        run_check_ocr()
    elif args.check_gemini:
        run_check_gemini()
    elif args.list_models:
        run_list_models()
    elif args.input:
        run_ingestion(args.input, recursive=args.recursive, force=args.force)
    elif args.ui:
        # Explicit UI flag – delegate to Streamlit runtime.
        import subprocess
        subprocess.run(["streamlit", "run", "user.py"])
    # If no specific flags are set, launch the Streamlit UI by default.
    elif not any([args.check_ocr, args.check_gemini, args.list_models, args.input, args.ui, args.filter is not None]):
        run_streamlit_app()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


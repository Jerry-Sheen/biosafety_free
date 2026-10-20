import os
"""
Unified Entrypoint Hub (app.py).
Allows running both User and Admin modules from a unified launcher,
while admin.py and user.py can also be executed completely independently.

Usage:
  - Unified Hub: streamlit run app.py
  - Admin Only:  streamlit run admin.py
  - User Only:   streamlit run user.py
"""
import streamlit as st
import admin
import user

st.set_page_config(
    page_title="RAG Vibe - Multi-document Local RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mode Selector
selected_app = st.sidebar.radio(
    "🧭 시스템 진입 모드 선택",
    ["💬 사용자 대화 (User Chat)", "🛠️ 관리자 문서 관리 (Admin Management)"],
    index=0
)

st.sidebar.divider()
st.sidebar.caption("💡 각 서비스는 개별 포트에서 독립 실행할 수도 있습니다:\n- `streamlit run user.py`\n- `streamlit run admin.py`")

if "사용자 대화" in selected_app:
    user.main()
else:
    admin.main()

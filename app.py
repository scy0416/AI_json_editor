from typing import TypedDict, List, Optional, Dict, Any

import streamlit as st

st.set_page_config(page_title="인공지능 JSON 에디터", layout="wide")
st.title("🧩 인공지능 JSON 에디터")
st.caption("자연어 지시 → RFC6902 패치 생성 → JSON Patch 도구 적용")

# 상태 그래프를 위한 상태
class AppState(TypedDict, total=False):
    instruction: str                # 사용자의 지시
    src: Dict[str, Any]             # 원본 JSON
    patch_ops: List[Dict[str, Any]] # JSON 패치 연산들
    result: Dict[str, Any]          # 패치 적용 결과
    error: Optional[str]            # 에러 메시지
    debug: Dict[str, Any]           # 디버그
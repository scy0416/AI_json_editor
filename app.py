import json, re, traceback
from typing import TypedDict, List, Optional, Dict, Any

import streamlit as st

from langchain_openai import ChatOpenAI

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

# 코드 블럭에서 JSON 추출 함수
def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(f"\s*```$", "", text)
    return text.strip()

# 딕서너리를 문자열로 반환
def safe_json_loads(s: str) -> Any:
    return json.loads(s)

# 생성된 패치 연산의 무결성 검사 함수
def validate_patch_ops(ops: Any) -> Optional[str]:
    if not isinstance(ops, list):
        return "패치가 배열(JSON Patch operations array)가 아닙니다."
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or "op" not in op or "path" not in op:
            return f"{i}번째 연산이 잘못되었습니다: 최소 'op'와 'path'가 필요합니다."
        if op["op"] in ("add", "replace", "test") and "value" not in op:
            return f"{i}번째 '{op['op']}' 연산에 'value'가 없습니다."
        if op["op"] in ("move", "copy") and "from" not in op:
            return f"{i}번째 '{op['op']}' 연산에 'from'이 없습니다."
    return None

# 사이트바 및 llm 설정
with st.sidebar:
    st.header("LLM 모델")
    model_name = st.selectbox(
        "사용할 LLM 모델",
        (
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-nano-2025-04-14"
        ),
        index=1
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

# llm생성 함수
def build_llm():
    return ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model=model_name,
        temperature=temperature
    )
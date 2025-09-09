import json, re, traceback
from typing import TypedDict, List, Optional, Dict, Any

import streamlit as st

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import SystemMessage
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

# 시스템 프롬프트
system_prompt = (
    "너는 RFC 6902(JSON Patch) 전문가다. 사용자의 지시를 바탕으로 "
    "주어진 '현재 JSON'을 수정하기 위한 JSON Patch 연산 배열만 정확히 출력해라. "
    "설명이나 코드펜스 없어 오직 유효한 JSON 배열만 출력한다. "
    "경로는 RFC 6901(JSON Pointer)을 따르고, 필요한 경우 특수문자 이스케이프(~0, ~1)을 사용하라. "
    "가능한 최소 연산으로 작성하고, 배열 끝 추가 시 /- 를 사용한다."
)

# 사용자 프롬프트
human_prompt = """
사용자 지시: {instruction}

현재 JSON:
```json
{src_json}
```

출력 형식: **RFC6902 operations array만**
예: [{"op":"replace","path":"/a","value":1}]
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(system_prompt),
    HumanMessagePromptTemplate.from_template(human_prompt)
])

# json 패치 생성 노드
def generate_patch(state: AppState) -> AppState:
    try:
        llm = build_llm()
        if llm is None:
            return {"error": "API 키가 없어 패치를 생성할 수 없습니다."}

        src_json = json.dumps(state["src"], ensure_ascii=False, indent=2)
        chain = prompt_template | llm
        resp = chain.invoke({"instruction": state["instruction"], "src_json": src_json})
        raw = resp.content or ""
        cleaned = strip_code_fences(raw)

        ops = safe_json_loads(cleaned)
        err = validate_patch_ops(ops)
        if err:
            return {"error": f"패치 유효성 오류: {err}", "debug": {"raw": raw, "cleaned": cleaned}}

        return {"patch_ops": ops, "debug": {"raw": raw, "cleaned": cleaned}}
    except Exception as e:
        return {"error": f"패치 생성 실패: {e}", "debug": {"trace": traceback.format_exc()}}

# 판단 노드
def judge(state: AppState) -> AppState:
    if state.get("error"):
        return state
    if not state.get("patch_ops"):
        return {"error": "생성된 패치가 없습니다."}
    return state

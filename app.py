import json, re, traceback
from typing import TypedDict, List, Optional, Dict, Any

import streamlit as st

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

import jsonpatch
from jsonpointer import JsonPointerException
from jsonpatch import JsonPatchConflict

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

# json 패치 적용 노드
def apply_patch(state: AppState) -> AppState:
    try:
        patched = jsonpatch.apply_patch(
            state["src"],
            state["patch_ops"],
            in_place=False
        )
        return {"result": patched}
    except (JsonPatchConflict, JsonPointerException, ValueError) as e:
        return {"error": f"패치 적용 실패: {e}"}
    except Exception as e:
        return {"error": f"알 수 없는 적용 오류: {e}", "debug": {"trace": traceback.format_exc()}}

# 그래프 생성 함수
def build_graph():
    g = StateGraph(AppState)
    g.add_node("패치 생성", generate_patch)
    g.add_node("판단", judge)
    g.add_node("JSON 패치 적용", apply_patch)

    g.add_edge(START, "패치 생성")
    g.add_edge("패치 생성", "판단")

    # 판단 결과 분기 함수
    def _route(s: AppState):
        if s.get("error"):
            return "end"
        if s.get("patch_ops"):
            return "apply"
        return "end"

    g.add_conditional_edges(
        "판단",
        _route,
        {
            "apply": "JSON 패치 적용",
            "end": END
        }
    )
    g.add_edge("JSON 패치 적용", END)
    return g.compile()

# 그래프 생성
graph = build_graph()

# streamlit UI 영역
left, right = st.columns([0.5, 0.5])

with left:
    st.subheader("1) 원본 JSON")
    default_json = {
        "name": "Alice",
        "age": 20,
        "tags": ["x", "y"],
        "profile": {"city": "Seoul"}
    }
    uploaded = st.file_uploader("JSON 업로드(옵션)", type=["json"])
    if "src_text" not in st.session_state:
        st.session_state.src_text = json.dumps(default_json, ensure_ascii=False, indent=2)

    if uploaded:
        st.session_state.src_text = uploaded.read().decode("utf-8")

    src_text = st.text_area(
        "편집 가능",
        st.session_state.src_text,
        height=260
    )

    st.subheader("2) 자연어 지시")
    instruction = st.text_area("예) 이름을 Bob으로 바꾸고, tags 끝에 'z'를 추가하고, age를 profile/age로 이동해줘.",
                               height=120,
                               value="이름을 Bob으로 바꾸고, tags 끝에 'z'를 추가하고, age를 profile/age로 이동해줘.")

    run = st.button("🚀 패치 생성 & 적용")

with right:
    st.subheader("실행 결과")

    if run:
        # 입력 파싱
        try:
            src_obj = json.loads(src_text)
        except Exception as e:
            st.error(f"원본 JSON 파싱 실패: {e}")
            src_obj = None

        # 파싱에 성공한 경우
        if src_obj is not None:
            init_state: AppState = {
                "instruction": instruction,
                "src": src_obj,
            }
            out: AppState = graph.invoke(init_state)

            # 디버그
            with st.expander("Debug (원시 모델 응답/정리본/trace)"):
                st.json(out.get("debug") or {})

            # 에러 처리
            if out.get("error"):
                st.error(out["error"])
            else:
                st.success("패치 생성 및 적용 완료!")

            # 생성된 패치
            st.subheader("생성된 JSON 패치")
            st.code(json.dumps(out.get("patch_ops", []), ensure_ascii=False, indent=2), language="json")

            # 적용 결과
            st.subheader("적용 결과 JSON")
            st.code(json.dumps(out.get("result", {}), ensure_ascii=False, indent=2), language="json")

            # 다운로드 버튼
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "⬇️ 패치(JSON) 다운로드",
                    data=json.dumps(out.get("patch_ops", []), ensure_ascii=False, indent=2),
                    file_name="patch.json",
                    mime="application/json"
                )
            with col_b:
                st.download_button(
                    "⬇️ 결과(JSON) 다운로드",
                    data=json.dumps(out.get("result", {}), ensure_ascii=False, indent=2),
                    file_name="result.json",
                    mime="application/json"
                )
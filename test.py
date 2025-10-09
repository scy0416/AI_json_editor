import json
from typing import TypedDict, List, Optional, Dict, Any
import streamlit as st
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage

class AppState(TypedDict, total=False):
    instruction: str
    src: Dict[str, Any]
    ops: List[Dict[str, Any]]
    result: Dict[str, Any]
    error: Optional[str]
    debug: Dict[str, Any]

st.set_page_config(page_title="인공지능 JSON 에디터(MCP활용)", layout="wide")
st.title("자연어 지시 → RFC6902 패치 생성 → JSON Patch 도구 적용")

with st.sidebar:
    st.header("MCP 서버")
    st.write("MCP URL: https://json-edit-mcp.de.r.appspot.com/mcp/")

    st.header("LLM 파라미터(서버에 그대로 전달)")
    model = st.selectbox(
        "LLM 모델",
        ("gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"),
        index=1,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

client = MultiServerMCPClient(
    {
        "json editor": {
            "url": "https://json-edit-mcp.de.r.appspot.com/mcp/",
            "transport": "streamable_http",
        }
    }
)
tools = asyncio.run(client.get_tools())
#print(tools[0].name)
json_agent = create_react_agent("openai:gpt-4.1", tools)

# json patch를 만드는 노드
def generate_json_patch(state: AppState):
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage("""너는 JSON 편집 전문가로 현재 JSON인 src에 사용자의 지시인 instruction을 적용하기 위한 패치를 도구를 사용해서 생성하라.
generate_json_patch를 호출하고, [instruction, src, model, temperature]를 전달해라.
도구의 결과를 정리하지 말고 그대로 반환해라."""),
        HumanMessagePromptTemplate.from_template("""### src:
{src}
### instruction
{instruction}""")
    ])
    prompt = prompt_template.invoke({"src": state["src"], "instruction": state["instruction"]})
    res = asyncio.run(json_agent.ainvoke(prompt))
    #print(res["messages"][0])
    # for m in res["messages"]:
    #     print("===")
    #     print(m)
    #print(json.loads(res["messages"][-1].content)["ops"])
    state["ops"] = json.loads(res["messages"][-1].content)["ops"]
    state["debug"] = dict()
    state["debug"]["raw"] = json.loads(res["messages"][-1].content)["raw"]
    state["debug"]["cleaned"] = json.loads(res["messages"][-1].content)["cleaned"]
    return state

def validate_json_patch(state: AppState):
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""너는 JSON 편집 전문가로 현재 JSON 편집을 위한 ops가 만들어져 있고, 이 ops가 적절한지 검사를 해야해.
이 때, validate_json_patch를 호출하고, ops를 포함하라.
도구의 결과에서 'ok'의 결과만을 그대로 반환하라.
### ops
{ops}""")
    ])
    prompt = prompt_template.invoke({"ops": state["ops"]})
    #print(prompt)
    res = asyncio.run(json_agent.ainvoke(prompt))
    #print(res["messages"][-1].content)
    if res["messages"][-1].content == "true":
        return
    else:
        state["error"] = res["messages"][-1].content
        return state

def apply_json_patch(state: AppState):
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""너는 JSON 편집 전문가로 현재 JSON 편집을 위한 ops가 있고, 이를 원본 JSON인 src에 적용해야해.
이 때, apply_json_patch를 호출하고, [src, ops]를 포함하도록 해.
도구의 결과를 그대로 반환해.
### src
{src}
### ops
{ops}""")
    ])
    prompt = prompt_template.invoke({"src": state["src"], "ops": state["ops"]})
    res = asyncio.run(json_agent.ainvoke(prompt))
    state["result"] = json.loads(res["messages"][-1].content)
    return state

graph_builder = StateGraph(AppState)
graph_builder.add_node("패치 생성", generate_json_patch)
graph_builder.add_node("패치 검사", validate_json_patch)
graph_builder.add_node("패치 적용", apply_json_patch)

graph_builder.add_edge(START, "패치 생성")
graph_builder.add_edge("패치 생성", "패치 검사")
graph_builder.add_edge("패치 검사", "패치 적용")
graph_builder.add_edge("패치 적용", END)

graph = graph_builder.compile()
###########################
left, right = st.columns(2)

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
        try:
            src_obj = json.loads(src_text)
        except Exception as e:
            st.error(f"원본 JSON 파싱 실패: {e}")
            src_obj = None

        if src_obj is not None:
            init_state: AppState = {
                "instruction": instruction,
                "src": src_obj
            }
            out: AppState = graph.invoke(init_state)

            with st.expander("Debug (원시 모델 응답/정리본/trace)"):
                st.json(out.get("debug") or {})

            if out.get("error"):
                st.error(out["error"])
            else:
                st.success("패치 생성 및 적용 완료!")

            st.subheader("생성된 JSON 패치")
            st.code(json.dumps(out.get("ops", {}), ensure_ascii=False, indent=2), language="json")

            st.subheader("적용 결과 JSON")
            st.code(json.dumps(out.get("result", {}), ensure_ascii=False, indent=2), language="json")

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "⬇️ 패치(JSON) 다운로드",
                    data=json.dumps(out.get("ops", []), ensure_ascii=False, indent=2),
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
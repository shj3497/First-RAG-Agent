# -*- coding: utf-8 -*-
from langgraph.graph import END, StateGraph
from typing import List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from core.tools.rag_search import RagSearchTool
from core.base_agent import get_chat_model


# --- 1. Graph State 정의 ---
# 그래프의 각 노드를 거치면서 데이터가 저장되고 업데이트되는 상태 객체입니다.
# TypedDict를 사용하여 각 필드의 타입을 명확히 정의합니다.
class GraphState(TypedDict):
    question: str  # 사용자의 원본 질문
    documents: List[str]  # RAG에서 검색된 문서 목록
    generation: str  # LLM이 생성한 답변
    grade: str  # 답변 평가 결과 (useful / not useful)
    iterations: int  # 재시도 횟수 (무한 루프 방지용)


# --- 2. 노드(Node) 함수 정의 ---
# 각 노드는 그래프의 한 단계를 나타내며, 특정 작업을 수행하는 함수입니다.

async def retrieve_documents_node(state: GraphState):
    """
    사용자의 질문을 받아 RAG 검색을 수행하여 관련 문서를 가져오는 노드입니다.
    """
    print("--- 📄 1. 문서 검색 ---")
    question = state["question"]
    iterations = state.get("iterations", 0)
    print(f"🔍 검색 질문: '{question}' (시도 횟수: {iterations + 1})")

    rag_tool = RagSearchTool()
    # execute 함수의 반환값이 (str, int) 튜플이므로, 두 변수로 받습니다.
    documents_string, doc_count = await rag_tool.execute(query=question)

    print(f"📚 검색된 최종 문서 수: {doc_count} 개")
    # LangGraph의 다음 노드로 포맷팅된 문자열을 리스트에 담아 전달합니다.
    return {"documents": [documents_string], "question": question, "iterations": iterations + 1}


async def generate_answer_node(state: GraphState):
    """
    검색된 문서를 바탕으로 LLM이 답변을 생성하는 노드입니다.
    """
    print("--- ✍️ 2. 답변 생성 ---")
    question = state["question"]
    documents = state["documents"]

    # 프롬프트를 정의합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 질문-답변(Question-Answering)을 수행하는 AI 어시턴트입니다. "
         "제공된 문서를 바탕으로 사용자의 질문에 대해 명확하고 간결하게 답변해주세요."),
        ("user",
         f"## 문서:\n\n---\n\n{{documents}}\n\n---\n\n## 질문:\n{{question}}")
    ])

    # LLM 모델을 정의합니다.
    llm = get_chat_model()

    # 프롬프트와 LLM을 연결합니다(LCEL).
    chain = prompt | llm

    # LLM을 호출하여 답변을 생성합니다.
    generation = await chain.ainvoke(
        {"documents": "\n\n".join(documents), "question": question})
    print(f"💬 생성된 답변: {generation.content[:100]}...")

    return {"generation": generation.content}


async def grade_answer_node(state: GraphState):
    """
    생성된 답변이 질문과 관련성이 있는지, 문서 내용을 잘 반영했는지를 평가하는 노드입니다.
    """
    print("--- 🤔 3. 답변 평가 ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 평가를 위한 프롬프트를 수정합니다.
    # '유용성'이라는 모호한 기준 대신, 답변이 '정답'을 포함하는지를 명확히 묻도록 변경합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 답변을 평가하는 AI 심사관입니다. "
         "주어진 '사용자 질문'에 대한 정답이 '검색된 문서'에 포함되어 있는지 확인해야 합니다. "
         "그 후, '생성된 답변'이 문서 내용을 바탕으로 질문에 대한 정답을 정확하게 제공하는지 평가해주세요. "
         "답변이 질문에 대한 정답을 명확히 포함하고 있다면 'yes', 그렇지 않거나 정보를 찾지 못했다고 답변하면 'no'라고만 평가해야 합니다."),
        ("user",
         f"## 검토 정보\n\n"
         f"### 검색된 문서:\n{''.join(documents)}\n\n"
         f"### 사용자 질문:\n{question}\n\n"
         f"### 생성된 답변:\n{generation}\n\n"
         "## 평가\n\n생성된 답변은 검색된 문서를 기반으로 사용자 질문에 대한 정답을 포함하고 있습니까? (yes / no)")
    ])
    llm = get_chat_model()
    chain = prompt | llm

    grade_result = await chain.ainvoke({})
    grade = grade_result.content
    print(f"📝 평가 결과: {grade}")

    # 평가 기준을 'yes'가 포함되었는지로 변경합니다.
    if "yes" in grade.lower():
        return {"grade": "useful"}  # 'useful' 상태는 그대로 유지하여 기존 흐름과 맞춥니다.
    else:
        return {"grade": "not useful"}


async def rewrite_question_node(state: GraphState):
    """
    답변의 점수가 낮을 경우, 더 나은 검색 결과를 얻기 위해 질문을 재작성하는 노드입니다.
    """
    print("--- 🔄 4. 질문 재작성 ---")
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 사용자 질문을 더 나은 검색어(search query)로 변환해주는 AI 어시턴트입니다."
         "원래 질문의 핵심 의도는 유지하면서, RAG 검색 시스템이 더 관련성 높은 문서를 찾을 수 있도록 질문을 재구성해주세요."
         "재구성된 질문만 간결하게 반환해주세요."),
        ("user", f"## 원래 질문:\n{question}\n\n"
                 "이 질문을 RAG 검색에 더 적합하도록 재구성해주세요.")
    ])
    llm = get_chat_model()
    chain = prompt | llm

    rewritten_question = await chain.ainvoke({})
    print(f"✨ 재작성된 질문: {rewritten_question.content}")

    return {"question": rewritten_question.content}


# --- 3. 조건부 엣지(Edge) 로직 ---
# 특정 노드를 실행한 후, 어떤 노드로 이동할지 결정하는 함수입니다.

def should_continue(state: GraphState):
    """
    답변 평가 결과에 따라 워크플로우를 계속할지, 아니면 종료할지 결정합니다.
    무한 루프를 방지하기 위해 최대 시도 횟수를 제한합니다.
    """
    grade = state.get("grade")
    iterations = state.get("iterations", 0)

    if iterations > 3:
        print("--- ⚠️ 최대 재시도 횟수 초과 ---")
        return "end"

    if grade == "useful":
        print("--- ✅ 답변이 유용하여 워크플로우 종료 ---")
        return "end"
    else:
        print("--- ❌ 답변이 유용하지 않아 질문 재작성 시도 ---")
        return "continue"


# --- 4. 그래프 빌드 ---
# langgraph를 사용하여 워크플로우 그래프를 구성합니다.

workflow = StateGraph(GraphState)

# 노드들을 그래프에 추가합니다.
workflow.add_node("retrieve", retrieve_documents_node)
workflow.add_node("generate", generate_answer_node)
workflow.add_node("grade", grade_answer_node)
workflow.add_node("rewrite", rewrite_question_node)

# 엣지들을 그래프에 추가하여 노드 간의 흐름을 정의합니다.
workflow.set_entry_point("retrieve")  # 시작점 설정
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade")

# 조건부 엣지를 추가합니다.
# 'grade' 노드 실행 후 'should_continue' 함수의 결과에 따라 분기됩니다.
# - "continue"를 반환하면 "rewrite" 노드로 이동합니다.
# - "end"를 반환하면 워크플로우가 종료(END)됩니다.
workflow.add_conditional_edges(
    "grade",
    should_continue,
    {
        "continue": "rewrite",
        "end": END
    }
)
workflow.add_edge("rewrite", "retrieve")  # 재작성 후 다시 검색 단계로 순환

# 그래프를 컴파일하여 실행 가능한 객체를 생성합니다.
app = workflow.compile()


# --- 5. 실행 함수 ---
async def run_langgraph_agent(user_query: str, session_id: str = None) -> str:
    """
    사용자 질문을 받아 langgraph로 구성된 에이전트를 실행하고 최종 답변을 반환합니다.
    (세션 ID는 현재 사용되지 않지만, `run_agent` 와의 호환성을 위해 유지합니다.)
    """
    inputs = {"question": user_query}
    final_state = await app.ainvoke(inputs)

    # 최종 상태에서 생성된 답변을 반환합니다.
    return final_state["generation"]


# 이 파일이 직접 실행될 때 테스트를 위한 코드입니다.
if __name__ == '__main__':
    # .env 파일 로드 (OPENAI_API_KEY)
    import asyncio
    from dotenv import load_dotenv
    import os

    # 현재 파일의 디렉토리의 부모 디렉토리를 기준으로 .env 파일을 찾습니다.
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    async def _test_run():
        query = "MCP 포털의 주요 기능은 무엇인가요?"
        print(f"🚀 LangGraph 에이전트 테스트 시작 (질문: '{query}')")
        final_answer = await run_langgraph_agent(query)
        print("\n\n---\n🏁 최종 답변:\n---")
        print(final_answer)

    asyncio.run(_test_run())

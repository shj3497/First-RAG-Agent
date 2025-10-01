# -*- coding: utf-8 -*-
from langgraph.graph import END, StateGraph
from typing import List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from core.tools.rag_search import RagSearchTool
from core.base_agent import get_chat_model
from core.history import get_history_store


# --- 0. 상수 정의 ---
# 대화 기록 컨텍스트 윈도우의 최대 메시지 수 (질문+답변)
# 짝수로 설정하는 것을 권장 (질문/답변 쌍)
MAX_CONVERSATION_HISTORY_MESSAGES = 20


# --- 1. Graph State 정의 ---
# 그래프의 각 노드를 거치면서 데이터가 저장되고 업데이트되는 상태 객체입니다.
# TypedDict를 사용하여 각 필드의 타입을 명확히 정의합니다.
class GraphState(TypedDict):
    question: str  # 사용자의 원본 질문
    documents: List[str]  # RAG에서 검색된 문서 목록
    generation: str  # LLM이 생성한 답변
    grade: str  # 답변 평가 결과 (useful / not useful)
    iterations: int  # 재시도 횟수 (무한 루프 방지용)
    chat_history: List[dict]  # 이전 대화 기록
    is_new_topic: bool  # 현재 질문이 새로운 주제인지 여부


# --- 2. 노드(Node) 함수 정의 ---
# 각 노드는 그래프의 한 단계를 나타내며, 특정 작업을 수행하는 함수입니다.

async def classify_topic_node(state: GraphState):
    """
    새로운 질문이 이전 대화의 주제와 이어지는지를 판단하는 노드입니다.
    """
    print("--- 🤔 0. 주제 분류 ---")
    question = state["question"]
    chat_history = state.get("chat_history", [])

    if not chat_history:
        print("💬 이전 대화 기록이 없어 새로운 주제로 판단합니다.")
        return {"is_new_topic": True}

    # 대화 기록을 간단한 문자열로 변환
    history_str = "\n".join(
        [f'{msg.get("role")}: {msg.get("content")}' for msg in chat_history]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 대화의 주제를 분류하는 AI입니다. "
         "주어진 '이전 대화'와 '새로운 질문'을 보고, 새로운 질문이 이전 대화의 주제를 이어가는지 아니면 완전히 새로운 주제인지 판단해주세요. "
         "답변은 'yes' (주제가 이어짐) 또는 'no' (새로운 주제) 로만 간결하게 해야 합니다."),
        ("user",
         f"## 이전 대화 (최대 {MAX_CONVERSATION_HISTORY_MESSAGES // 2}쌍):\n{history_str}\n\n"
         f"## 새로운 질문:\n{question}\n\n"
         "이 새로운 질문은 이전 대화의 주제와 관련이 있습니까? (yes / no)")
    ])
    llm = get_chat_model(temperature=0)
    chain = prompt | llm
    result = await chain.ainvoke({})

    if "no" in result.content.lower():
        print("💬 LLM이 새로운 주제로 판단했습니다.")
        return {"is_new_topic": True}
    else:
        print("💬 LLM이 기존 주제가 이어지는 것으로 판단했습니다.")
        return {"is_new_topic": False}


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
    chat_history = state.get("chat_history", [])
    is_new_topic = state.get("is_new_topic", True)

    # 프롬프트를 정의합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 질문-답변(Question-Answering)을 수행하는 AI 어시턴트입니다. "
         "제공된 문서와 이전 대화 내용을 바탕으로 사용자의 질문에 대해 명확하고 간결하게 답변해주세요."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user",
         "## 문서:\n\n---\n\n{documents}\n\n---\n\n## 질문:\n{question}")
    ])

    # LLM 모델을 정의합니다.
    llm = get_chat_model()

    # 프롬프트와 LLM을 연결합니다(LCEL).
    chain = prompt | llm

    # 새로운 주제인 경우, 대화 기록을 비워서 전달합니다.
    effective_history = chat_history if not is_new_topic else []

    # 대화 기록의 형식을 LangChain 모델이 이해할 수 있는 형태로 변환합니다.
    history_messages = []
    for msg in effective_history:
        if msg.get("role") == "user":
            history_messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            history_messages.append(AIMessage(content=msg.get("content")))

    # LLM을 호출하여 답변을 생성합니다.
    generation = await chain.ainvoke({
        "documents": "\n\n".join(documents),
        "question": question,
        "chat_history": history_messages
    })
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
    chat_history = state.get("chat_history", [])
    is_new_topic = state.get("is_new_topic", True)

    # 새로운 주제이거나 대화 기록이 없으면, 현재 질문만으로 재작성합니다.
    if is_new_topic:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 사용자 질문을 더 나은 검색어(search query)로 변환해주는 AI 어시턴트입니다."
             "원래 질문의 핵심 의도는 유지하면서, RAG 검색 시스템이 더 관련성 높은 문서를 찾을 수 있도록 질문을 재구성해주세요."
             "재구성된 질문만 간결하게 반환해주세요."),
            ("user", f"## 원래 질문:\n{question}\n\n"
                     "이 질문을 RAG 검색에 더 적합하도록 재구성해주세요.")
        ])
    # 기존 주제가 이어지는 경우, 대화 기록을 함께 사용합니다.
    else:
        # 대화 기록을 문자열로 변환하여 프롬프트에 포함합니다.
        history_str = "\n".join(
            [f'{msg.get("role")}: {msg.get("content")}' for msg in chat_history]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 사용자 질문을 더 나은 검색어(search query)로 변환해주는 AI 어시턴트입니다."
             "원래 질문의 핵심 의도는 유지하면서, 이전 대화의 맥락을 참고하여 RAG 검색 시스템이 더 관련성 높은 문서를 찾을 수 있도록 질문을 재구성해주세요."
             "재구성된 질문만 간결하게 반환해주세요."),
            ("user", f"## 이전 대화:\n{history_str}\n\n"
                     f"## 현재 질문:\n{question}\n\n"
                     "이전 대화와 현재 질문을 바탕으로 RAG 검색에 가장 적합한 질문을 하나로 재구성해주세요.")
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
workflow.add_node("classify_topic", classify_topic_node)
workflow.add_node("retrieve", retrieve_documents_node)
workflow.add_node("generate", generate_answer_node)
workflow.add_node("grade", grade_answer_node)
workflow.add_node("rewrite", rewrite_question_node)

# 엣지들을 그래프에 추가하여 노드 간의 흐름을 정의합니다.
workflow.set_entry_point("classify_topic")  # 시작점 변경
workflow.add_edge("classify_topic", "retrieve")
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
    session_id를 사용하여 대화 기록을 관리합니다.
    """
    # 1. 대화 기록 관리자 및 이전 기록 로드
    history = get_history_store()
    past_messages = []
    if session_id:
        print(
            f"🧠 세션 '{session_id}'에서 이전 대화 기록을 로드합니다 (최대 {MAX_CONVERSATION_HISTORY_MESSAGES}개).")
        past_messages = history.get_messages(session_id)[
            -MAX_CONVERSATION_HISTORY_MESSAGES:]

    # 2. 그래프 실행을 위한 입력값 구성
    # is_new_topic은 첫 노드에서 결정되므로 여기서 초기화할 필요가 없습니다.
    inputs = {"question": user_query, "chat_history": past_messages}

    # 3. LangGraph 에이전트 실행
    final_state = await app.ainvoke(inputs)
    final_answer = final_state["generation"]

    # 4. 대화 기록 저장
    if session_id:
        new_messages = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": final_answer}
        ]
        history.add_messages(session_id, new_messages)
        print(
            f"💾 세션 '{session_id}'에 대화 기록 {len(new_messages)}개 저장 완료")

    # 5. 최종 답변 반환
    return final_answer


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

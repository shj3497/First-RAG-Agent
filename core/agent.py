# core/agent.py
import openai
import json
from typing import List
from openai import AsyncOpenAI

from core.tools.base import Tool
from core.tools.rag_search import RagSearchTool
from core.history import get_history_store  # 대화 기록 관리 모듈 import

# --- 에이전트 설정 ---
# 이 에이전트가 사용할 수 있는 도구들의 리스트입니다.
# 지금은 RagSearchTool 하나만 있지만, 나중에 다른 도구들을 추가할 수 있습니다.
AVAILABLE_TOOLS: List[Tool] = [
    RagSearchTool(),
]

# 사용할 OpenAI 모델을 지정합니다. Tool Calling을 지원하는 최신 모델을 사용하는 것이 좋습니다.
AGENT_MODEL = "gpt-4-turbo"

# OpenAI 비동기 클라이언트는 지연 초기화합니다.
_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


async def run_agent(user_query: str, session_id: str = None) -> str:
    """
    사용자의 질문을 받아 AI 에이전트의 전체 로직을 실행하고 최종 답변을 반환합니다.
    대화 기록이 있는 경우(session_id 제공 시) 이를 포함하여 맥락을 유지합니다.
    OpenAI의 Tool Calling 기능을 사용하여 적절한 도구를 선택하고 실행합니다.
    """
    print(f"🤖 에이전트 실행 시작 (세션 ID: {session_id}, 질문: '{user_query}')")

    # 대화 기록 관리자를 가져옵니다.
    history = get_history_store()

    # --- 1. 메시지 목록 구성 ---
    # 시스템 프롬프트는 항상 대화의 시작에 위치합니다.
    system_prompt = {"role": "system",
                     "content": "You are a helpful assistant that can use tools to answer questions."}

    # 세션 ID가 있을 경우, 이전 대화 기록을 가져옵니다.
    if session_id:
        past_messages = history.get_messages(session_id)
    else:
        past_messages = []

    # 현재 사용자 메시지를 생성합니다.
    user_message = {"role": "user", "content": user_query}

    # 최종적으로 LLM에게 보낼 전체 메시지 목록을 구성합니다.
    # [시스템 프롬프트, 이전 대화들, 현재 질문]
    messages = [system_prompt] + past_messages + [user_message]

    # 이번 턴(turn)에서 대화 기록에 추가할 메시지들을 담을 리스트입니다.
    new_messages_for_history = [user_message]

    # --- 2. 1차 LLM 호출: 도구 사용 결정 ---
    tools_for_openai = [tool.to_openai_format() for tool in AVAILABLE_TOOLS]

    print("🧠 1차 호출: LLM에게 도구 선택 요청...")
    first_response = await _get_openai_client().chat.completions.create(
        model=AGENT_MODEL,
        messages=messages,  # 전체 대화 기록을 포함하여 전달
        tools=tools_for_openai,
        tool_choice="auto",  # LLM이 도구 사용 여부를 자동으로 결정하도록 합니다.
        temperature=0,  # 답변의 일관성을 위해 0으로 설정
    )

    response_message = first_response.choices[0].message
    tool_calls = response_message.tool_calls

    # LLM의 응답(도구 호출 요청)도 기록 대상입니다.
    # OpenAI 라이브러리의 응답 객체를 나중에 직렬화 가능한 dict 형태로 변환하여 추가합니다.
    new_messages_for_history.append(
        response_message.model_dump(exclude_unset=True))

    # --- 3. LLM의 응답 분석 및 도구 실행 ---
    if tool_calls:
        print(f"✅ LLM이 도구 사용 결정: {len(tool_calls)}개")

        # 병렬 도구 호출을 지원하지만, 여기서는 첫 번째 도구 호출만 처리하는 것으로 단순화합니다.
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # 사용 가능한 도구 목록에서 LLM이 선택한 도구를 찾습니다.
        tool_to_use = next(
            (tool for tool in AVAILABLE_TOOLS if tool.name == tool_name), None)

        if not tool_to_use:
            return f"오류: LLM이 알 수 없는 도구 '{tool_name}'을 호출했습니다."

        # 선택된 도구의 execute 메서드를 호출하여 실제 로직을 실행합니다.
        print(f"🏃 도구 실행: {tool_name}({tool_args})")
        tool_output = await tool_to_use.execute(**tool_args)

        # --- 4. 2차 LLM 호출: 도구 결과로 최종 답변 생성 ---
        # 도구 실행 결과를 API에 전달하기 위한 메시지를 구성합니다.
        tool_message = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_name,
            "content": tool_output,
        }
        # 도구 실행 결과 메시지도 기록 대상입니다.
        new_messages_for_history.append(tool_message)

        # 2차 호출을 위해 전체 대화 흐름을 다시 만듭니다.
        messages_for_second_call = messages + \
            [response_message.model_dump(exclude_unset=True), tool_message]

        print("🧠 2차 호출: 도구 결과를 바탕으로 최종 답변 생성 요청...")
        second_response = await _get_openai_client().chat.completions.create(
            model=AGENT_MODEL,
            messages=messages_for_second_call,
            temperature=0,  # 답변의 일관성을 위해 0으로 설정
        )

        final_answer = second_response.choices[0].message.content

        # 2차 호출 응답(최종 답변) 메시지도 기록 대상입니다.
        final_message = second_response.choices[0].message.model_dump(
            exclude_unset=True)
        new_messages_for_history.append(final_message)

        print("✅ 최종 답변 생성 완료")

    else:
        # --- 5. 도구 미사용 시 ---
        #    첫 번째 응답에 포함된 일반 답변을 그대로 반환합니다.
        print("✅ LLM이 도구 없이 직접 답변 결정")
        final_answer = response_message.content
        # 이 경우 `response_message`가 최종 답변이므로, 이미 기록 대상에 추가되었습니다.

    # --- 6. 대화 기록 저장 ---
    # 세션 ID가 제공된 경우에만 이번 턴의 대화 내용을 기록에 추가합니다.
    if session_id:
        history.add_messages(session_id, new_messages_for_history)
        print(
            f"💾 세션 '{session_id}'에 대화 기록 {len(new_messages_for_history)}개 저장 완료")

    return final_answer

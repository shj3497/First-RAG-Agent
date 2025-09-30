# core/tool_calling_agent.py
# -*- coding: utf-8 -*-
import json
from typing import List

from core.history import get_history_store
from core.base_agent import AVAILABLE_TOOLS, AGENT_MODEL, get_openai_client


async def run_tool_calling_agent(user_query: str, session_id: str = None) -> str:
    """
    OpenAI의 Tool Calling 기능을 사용하는 에이전트 로직입니다.
    LLM이 스스로 판단하여 RAG 검색 도구를 사용할지 결정합니다.
    """
    print(
        f"🤖 (Tool Calling Agent) 실행 시작 (세션 ID: {session_id}, 질문: '{user_query}')")

    # 대화 기록 관리자를 가져옵니다.
    history = get_history_store()

    # --- 1. 메시지 목록 구성 ---
    system_prompt = {"role": "system",
                     "content": "You are a helpful assistant that can use tools to answer questions."}

    if session_id:
        past_messages = history.get_messages(session_id)
    else:
        past_messages = []

    user_message = {"role": "user", "content": user_query}
    messages = [system_prompt] + past_messages + [user_message]
    new_messages_for_history = [user_message]

    # --- 2. 1차 LLM 호출: 도구 사용 결정 ---
    tools_for_openai = [tool.to_openai_format() for tool in AVAILABLE_TOOLS]

    print("🧠 1차 호출: LLM에게 도구 선택 요청...")
    first_response = await get_openai_client().chat.completions.create(
        model=AGENT_MODEL,
        messages=messages,
        tools=tools_for_openai,
        tool_choice="auto",
        temperature=0,
    )

    response_message = first_response.choices[0].message
    tool_calls = response_message.tool_calls
    new_messages_for_history.append(
        response_message.model_dump(exclude_unset=True))

    # --- 3. LLM의 응답 분석 및 도구 실행 ---
    if tool_calls:
        print(f"✅ LLM이 도구 사용 결정: {len(tool_calls)}개")
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        tool_to_use = next(
            (tool for tool in AVAILABLE_TOOLS if tool.name == tool_name), None)

        if not tool_to_use:
            return f"오류: LLM이 알 수 없는 도구 '{tool_name}'을 호출했습니다."

        print(f"🏃 도구 실행: {tool_name}({tool_args})")
        tool_output = await tool_to_use.execute(**tool_args)

        # --- 4. 2차 LLM 호출: 도구 결과로 최종 답변 생성 ---
        tool_message = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_name,
            "content": tool_output,
        }
        new_messages_for_history.append(tool_message)

        messages_for_second_call = messages + \
            [response_message.model_dump(exclude_unset=True), tool_message]

        print("🧠 2차 호출: 도구 결과를 바탕으로 최종 답변 생성 요청...")
        second_response = await get_openai_client().chat.completions.create(
            model=AGENT_MODEL,
            messages=messages_for_second_call,
            temperature=0,
        )
        final_answer = second_response.choices[0].message.content
        final_message = second_response.choices[0].message.model_dump(
            exclude_unset=True)
        new_messages_for_history.append(final_message)
        print("✅ 최종 답변 생성 완료")

    else:
        # --- 5. 도구 미사용 시 ---
        print("✅ LLM이 도구 없이 직접 답변 결정")
        final_answer = response_message.content

    # --- 6. 대화 기록 저장 ---
    if session_id:
        history.add_messages(session_id, new_messages_for_history)
        print(
            f"💾 세션 '{session_id}'에 대화 기록 {len(new_messages_for_history)}개 저장 완료")

    return final_answer

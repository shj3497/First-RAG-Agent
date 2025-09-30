# core/agent.py
# LangGraph 에이전트를 사용하기 위해 import 합니다.
from core.langgraph_agent import run_langgraph_agent
# ToolCalling 에이전트를 사용하기 위해 import 합니다.
from core.tool_calling_agent import run_tool_calling_agent


async def run_agent(user_query: str, session_id: str = None) -> str:
    """
    사용자의 질문을 받아 AI 에이전트의 전체 로직을 실행하고 최종 답변을 반환합니다.

    아래에서 사용하고 싶은 에이전트 로직의 주석을 해제하여 테스트할 수 있습니다.
    - run_tool_calling_agent: OpenAI Tool Calling 기반 에이전트
    - run_langgraph_agent: 답변 품질을 평가하고 질문을 재작성하는 LangGraph 기반 에이전트
    """
    # --- 실행할 에이전트 선택 ---
    # 사용하려는 에이전트의 주석을 해제하고, 다른 하나는 주석 처리해주세요.

    # 1. LangGraph 기반 에이전트 (답변 평가 및 질문 재작성 기능 포함)
    return await run_langgraph_agent(user_query, session_id)

    # 2. OpenAI Tool Calling 기반 에이전트
    # return await run_tool_calling_agent(user_query, session_id)

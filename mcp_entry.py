# mcp_entry.py
from core.agent import run_agent
from fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()


# 1. FastMCP 서버 인스턴스를 생성합니다.
#    - 'title' 인자는 존재하지 않으므로 삭제합니다.
#    - 'description' 대신 'instructions' 인자를 사용합니다.
mcp_server = FastMCP(
    name="MyCustomAgentServer",
    instructions="이 서버는 사용자의 질문에 답변하는 AI 에이전트입니다.",
)

# 2. 에이전트 실행 로직을 MCP '도구(Tool)'로 등록합니다.
#    - @mcp_server.tool() 데코레이터를 사용하여 함수를 도구로 만듭니다.
#    - 클라이언트(Cursor 등)는 이 도구의 이름('ask')을 호출하게 됩니다.


@mcp_server.tool(name="ask")
async def ask_agent(question: str) -> str:
    """
    사용자의 질문을 받아 AI 에이전트에게 전달하고 최종 답변을 반환합니다.
    """
    # run_agent 함수는 session_id를 받을 수 있지만, MCP Tool에서는 필수가 아니므로
    # user_query만 전달하여 호출합니다.
    answer = await run_agent(user_query=question)
    return answer

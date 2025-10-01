# server_sse.py
import uvicorn
from mcp_entry import mcp_server
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# MCP 서버로부터 SSE 통신 전용 ASGI 애플리-케이션을 생성합니다.
mcp_sse_app = mcp_server.http_app(path="/", transport="streamable-http")

if __name__ == "__main__":
    """
    이 스크립트는 SSE(Server-Sent Events) 방식을 통해 MCP 클라이언트와 통신하는
    웹 서버를 실행합니다. 서버 환경에서 Cursor와 연동할 때 사용됩니다.
    """
    # API 서버와의 충돌을 피하기 위해 다른 포트(예: 8001)를 사용합니다.
    print("서버 환경을 위한 sse 모드로 MCP 서버를 실행합니다. (http://127.0.0.1:8001)")
    uvicorn.run(mcp_sse_app, host="127.0.0.1", port=8001)

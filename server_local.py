# server_local.py
import asyncio
from mcp_entry import mcp_server
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

if __name__ == "__main__":
    """
    이 스크립트는 표준 입출력(stdio)을 통해 MCP 클라이언트와 통신하는
    로컬 전용 서버를 실행합니다. Cursor의 'stdio' 연결 방식에 사용됩니다.
    """
    print("로컬 개발 환경을 위한 stdio 모드로 MCP 서버를 실행합니다.")

    # mcp_server의 serve_stdio()를 직접 실행하여 통신을 시작합니다.
    # 이 함수는 비동기로 동작하므로 asyncio.run()을 사용합니다.
    asyncio.run(mcp_server.run())

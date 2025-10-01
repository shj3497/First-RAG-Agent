# main.py
import os
import uuid
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
from core.agent import run_agent
from core.rag_builder import build_rag_from_path
from dotenv import load_dotenv
import uvicorn

load_dotenv()


# --- mcp_entry에서 MCP 서버 인스턴스를 직접 임포트 ---


# --- 환경변수 로딩을 최상단으로 이동 ---
# 이제 환경변수가 설정된 상태에서 모듈이 임포트되므로 오류가 발생하지 않습니다.
# 에이전트의 메인 실행 함수를 가져옵니다.


# --- Pydantic 모델 정의 ---
# Pydantic 모델은 API 요청 및 응답의 데이터 형식을 정의하고 유효성을 검사하는 데 사용됩니다.
# 프론트엔드에서 TypeScript의 interface와 유사한 역할을 합니다.


class QueryRequest(BaseModel):
    """
    /query 엔드포인트의 요청 본문(body) 형식을 정의하는 모델입니다.
    """
    question: str  # 사용자가 입력한 질문
    sessionId: Optional[str] = None  # 사용자 세션을 식별하기 위한 선택적 ID


class RagBuildRequest(BaseModel):
    """
    /build-rag 엔드포인트의 요청 본문(body) 형식을 정의하는 모델입니다.
    콘텐츠를 스크래핑할 사이트의 URL을 전달받습니다.
    """
    site_url: str  # 예: "https://www.megazone.com" 또는 "http://localhost:3000"


# --- FastAPI 앱 초기화 ---
# FastAPI 인스턴스를 생성하여 웹 애플리케이션을 초기화합니다.
# 이 'app' 변수를 uvicorn이 실행하여 서버를 켭니다.

app = FastAPI(
    title="AI Agent Server API",
    version="1.0.0",
    description="""
메가존클라우드 내부 정보를 답변하는 AI 에이전트 서버입니다.
주요 기능:
- **RAG 기반 Q&A**: 내부 문서를 검색하여 질문에 답변합니다.
- **채팅 기능**: `sessionId`를 통해 대화의 맥락을 유지합니다.
- **MCP 연동**: Cursor와 같은 외부 도구와 연동할 수 있습니다.
    """,
)


# --- API 엔드포인트(라우터) 정의 ---

@app.post("/api/sessions")
def create_session():
    """
    새로운 대화 세션을 위한 고유 ID를 생성하여 반환합니다.
    웹 클라이언트는 채팅을 시작하기 전에 이 API를 호출하여 세션 ID를 발급받아야 합니다.
    """
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    사용자의 질문을 받아 AI 에이전트의 답변을 반환하는 메인 엔드포인트입니다.
    실제 로직은 core/agent.py의 run_agent 함수에 위임합니다.
    """
    # core/agent.py에 구현된 메인 에이전트 함수를 호출하고 결과를 반환합니다.
    final_answer = await run_agent(user_query=request.question, session_id=request.sessionId)
    return {"answer": final_answer}


@app.post("/build-rag")
async def build_rag_index(request: RagBuildRequest):
    """
    지정된 사이트의 sitemap.xml을 기반으로 RAG 데이터베이스를 구축/업데이트하는 엔드포인트입니다.
    헤드리스 브라우저를 사용하여 동적으로 렌더링된 콘텐츠까지 모두 수집합니다.
    """
    # core/rag_builder.py의 함수가 비동기로 변경되었으므로, `await` 키워드를 사용해 호출해야 합니다.
    result = await build_rag_from_path(request.site_url)
    return result


@app.get("/")
def read_root():
    """
    서버가 정상적으로 실행 중인지 확인하기 위한 기본 GET 엔드포인트입니다.
    브라우저에서 서버 주소로 접속했을 때 이 메시지가 보이면 서버가 켜진 것입니다.
    """
    return {"message": "AI Agent Server is running."}


if __name__ == "__main__":
    print("FastAPI 기반 REST API 서버를 실행합니다. (http://127.0.0.1:8000)")
    uvicorn.run(app, host="127.0.0.1", port=8000)

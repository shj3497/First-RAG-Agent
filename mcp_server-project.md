# 프로젝트 명세서: AI 에이전트 'MCP 서버' 백엔드 구축 (Python)

## 1. 프로젝트 개요

### 목표

사용자의 질문 의도를 파악하고, 사전 정의된 다양한 도구(Tools)를 선택적으로 사용하여 최적의 답변을 생성하는 지능형 AI 에이전트 API 서버를 구축한다.

### 핵심 기능

- **API 엔드포인트**: 외부 클라이언트(웹사이트 챗봇 등)로부터 질문을 받을 수 있는 API 엔드포인트를 제공한다.
- **의도 분석 및 도구 선택**: LLM을 사용하여 사용자의 질문을 분석하고, 가장 적합한 도구를 선택하는 라우팅(Routing) 기능을 구현한다.
- **도구 실행**: 선택된 도구의 로직을 실행하여 필요한 정보를 가져온다. (초기 단계에서는 RAG 검색 도구만 구현)
- **대화 맥락 유지 (채팅 기능)**: `session_id`를 기반으로 이전 대화 기록을 기억하여, 연속적인 질문에 대해 맥락을 이해하고 답변한다.
- **최종 답변 생성**: 도구 실행 결과를 바탕으로 LLM이 사용자 친화적인 최종 답변을 생성하여 반환한다.

### 기술 스택

- **언어**: **Python**
- **웹 프레임워크**: **FastAPI**
- **주요 라이브러리**:
  - `openai`: OpenAI의 LLM(GPT-4 등) 및 임베딩 모델 사용
  - `[Vector DB Client]`: 벡터 데이터베이스 클라이언트. 로컬 테스트 시 `chromadb`, `faiss-cpu` 사용 (예: `pinecone-client`, `weave-client`)
  - `python-dotenv`: 환경변수 관리
  - `uvicorn`: FastAPI 개발 서버 실행

---

## 2. API 명세

### 엔드포인트: `POST /api/sessions`

- **설명**: 새로운 대화 세션을 위한 고유 ID를 생성하여 반환합니다. 웹 클라이언트는 채팅을 시작하기 전에 이 API를 호출하여 세션 ID를 발급받아야 합니다.
- **요청 (Request)**: Body 없음
- **응답 (Response)**:
  ```json
  {
    "session_id": "새로 생성된 고유 세션 ID (UUID)"
  }
  ```

### 엔드포인트: `POST /query`

- **설명**: 사용자의 질문을 받아 AI 에이전트의 답변을 반환하는 메인 엔드포인트입니다. `sessionId`를 포함하여 요청하면 이전 대화 기록을 바탕으로 답변합니다.

#### 요청 (Request)

- **Header**:
  - `Content-Type: application/json`
- **Body**:
  ```json
  {
    "question": "사용자가 입력한 질문 텍스트",
    "sessionId": "사용자 세션을 식별하기 위한 고유 ID (옵션). `/api/sessions`를 통해 발급받은 값을 사용합니다."
  }
  ```

### 엔드포인트: `POST /build-rag`

- **설명**: 지정된 `site_url`의 `sitemap.xml`을 기반으로 RAG 데이터베이스를 구축하거나 업데이트합니다.
- **요청 (Request)**:
  ```json
  {
    "site_url": "콘텐츠를 수집할 사이트의 URL (예: https://www.megazone.com)"
  }
  ```
- **응답 (Response)**:
  ```json
  {
    "status": "success",
    "message": "RAG 인덱스 빌드 완료. 처리: N개, 변경 없음: M개, 삭제: K개."
  }
  ```

## 3. 세부 구현 요구사항

### Step 0: RAG 데이터 파이프라인 구축 (효율적인 Upsert 로직)

- **목표**: 웹사이트(`sitemap.xml` 기반)의 콘텐츠를 기반으로 RAG 데이터를 구축하되, 변경된 내용만 효율적으로 갱신하여 중복을 방지하고 리소스를 절약한다.

- **콘텐츠 추출 방식 (Suspense 대응)**:

  - `next build`로 생성된 정적 HTML 파일은 클라이언트 사이드 렌더링(CSR) 및 `Suspense`로 인한 콘텐츠 누락이 있을 수 있다.
  - 이를 해결하기 위해, 빌드 결과물을 로컬 서버로 실행한 뒤, **헤드리스 브라우저(예: Playwright, Puppeteer)를 사용**하여 각 페이지에 접속한다.
  - 페이지의 모든 JavaScript 실행과 데이터 로딩이 완료된 후의 **최종 렌더링된 HTML**을 추출하여 RAG 데이터 소스로 사용한다.

- **데이터 식별**:

  - **고유 ID**: 각 콘텐츠의 고유 식별자로 페이지의 URL 경로를 사용한다. (예: `/about`, `/posts/post-1`)
  - **콘텐츠 해시**: HTML에서 추출한 텍스트 본문의 내용으로 MD5 또는 SHA256 해시를 생성하여 '콘텐츠 지문(fingerprint)'으로 사용한다.

- **벡터 DB 스키마**:

  - 벡터 데이터를 저장할 때, 벡터 값과 함께 다음 메타데이터를 저장한다.
    - `page_url`: 고유 식별자
    - `content_hash`: 콘텐츠 내용의 해시값
    - `chunk_text`: 원본 텍스트 조각

- **구축 및 갱신 프로세스 (Next.js 빌드 시 트리거)**:
  1. Next.js 빌드 파이프라인에서 RAG 구축 API 엔드포인트를 호출한다.
  2. 스크립트가 빌드 결과물을 로컬 서버로 실행한다.
  3. **삭제된 페이지 처리**:
     a. 현재 빌드된 사이트맵 또는 라우트 목록에서 모든 페이지의 `page_url` 목록을 생성한다 (`current_pages`).
     b. 벡터 DB에 저장된 모든 페이지의 `page_url` 목록을 가져온다 (`db_pages`).
     c. `db_pages`에는 있지만 `current_pages`에는 없는 URL을 '삭제된 페이지'로 식별한다.
     d. 식별된 '삭제된 페이지'에 해당하는 모든 벡터를 DB에서 **삭제(DELETE)**한다.
  4. **신규/변경된 페이지 처리**:
     a. 스크립트는 `current_pages` 목록을 순회하며, 헤드리스 브라우저로 각 페이지에 접속하여 최종 렌더링된 HTML을 가져온다.
     b. 가져온 HTML에서 텍스트를 추출하고, `content_hash`를 계산한다.
     c. 벡터 DB에서 해당 `page_url`을 가진 데이터가 있는지 조회한다.
  5. **분기 처리**:
     - **CASE 1: 신규 페이지 (DB에 `page_url` 없음)**
       - 페이지 내용을 청크로 분할하고, 임베딩하여 메타데이터와 함께 벡터 DB에 **새로 저장(INSERT)**한다.
     - **CASE 2: 기존 페이지 (DB에 `page_url` 있음)**
       - DB에 저장된 `content_hash`와 새로 계산한 `content_hash`를 비교한다.
       - **(a) 해시 동일 (변경 없음)**: 아무 작업 없이 건너뛴다. (핵심적인 중복 방지)
       - **(b) 해시 다름 (콘텐츠 변경)**: 해당 `page_url`을 가진 기존 벡터를 모두 **삭제(DELETE)**한 후, 변경된 콘텐츠의 새로운 벡터를 **저장(INSERT)**한다.

### Step 1: 기본 서버 구조 설정

- FastAPI를 사용하여 기본 서버를 설정한다. (`main.py` 파일 생성)
- `POST /query` 경로 처리 함수(path operation function)를 생성한다.
- 요청 본문(body)의 유효성 검사를 위해 Pydantic 모델을 정의한다.
- 환경변수 관리를 위해 `.env` 파일을 사용하고, `python-dotenv` 라이브러리로 로드하도록 설정한다.
  - 필요한 환경변수: `OPENAI_API_KEY`, `VECTOR_DB_API_KEY`, `VECTOR_DB_ENVIRONMENT` 등

### Step 2: 도구(Tool) 정의 및 초기 구현

- 모든 도구가 상속받을 `Tool` 추상 기반 클래스(Abstract Base Class)를 Python으로 정의한다. 이 클래스는 다음을 포함해야 한다.
  - `name`: 도구의 이름 (예: `rag_search_tool`)
  - `description`: LLM이 언제 이 도구를 선택해야 하는지 판단할 근거가 되는 설명 (예: "회사 내부 문서, 제품 정보, 가이드라인에 대한 질문에 답변할 때 사용합니다.")
  - `execute`: 도구의 실제 로직을 실행하는 비동기 메서드.
- 초기 단계에서는 `rag_search_tool` 하나만 구현한다.
- 이 도구의 `execute` 메서드는 다음 로직을 포함한다:
  1. 입력받은 텍스트를 OpenAI 임베딩 API를 사용해 벡터로 변환한다.
  2. 변환된 벡터를 사용해 벡터 DB에서 유사도 높은 문서를 검색한다.
  3. 검색된 문서들의 텍스트를 종합하여 문자열로 반환한다.

### Step 3: 터미널 기반 대화형 RAG 테스트 도구

- 사용자가 터미널에서 직접 질문을 입력하고 RAG 검색 도구의 답변을 바로 확인할 수 있는 대화형 CLI(Command-Line Interface) 스크립트를 작성한다.
- 이 도구는 `rag_search_tool`의 성능을 API 서버와 분리하여 독립적으로 테스트하고 디버깅하는 데 사용된다.
- 스크립트는 내부적으로 `rag_search_tool`의 `execute` 메서드를 호출하여 결과를 가져와 사용자에게 보여준다.

### Step 4: FastMCP를 이용한 모듈화된 MCP 서버 구축

- **목표**: 기존 FastAPI 서버에 **FastMCP**를 통합하여 외부 AI 어시스턴트(Cursor 등)와 연동 가능한 MCP 서버 기능을 추가합니다. 이때, MCP 관련 코드는 `mcp_entry.py` 파일에 **모듈화**하여 관리하고, 메인 서버 실행 시(`uvicorn main:app`) 자동으로 통합되어 실행되도록 합니다.

- **핵심 로직**:

  1.  `mcp_entry.py` 파일에 MCP 엔드포인트들을 관리할 FastAPI `APIRouter`를 생성합니다.
  2.  생성된 라우터에 `FastMCP`를 사용하여 MCP 표준 규격에 맞는 엔드포인트(예: `/info`, `/request`)를 등록하고, 요청 처리 로직을 `core.agent.run_agent` 함수와 직접 연결합니다. 통신 방식은 SSE(Server-Sent Events)를 사용합니다.
  3.  `main.py` 파일에서 `mcp_entry.py`에 정의된 `APIRouter`를 임포트합니다.
  4.  메인 `FastAPI` 앱 인스턴스에 `app.include_router()`를 사용하여 임포트한 MCP 라우터를 등록합니다.
  5.  결과적으로 `uvicorn main:app --reload` 단일 명령어로 서버를 실행하면, `main.py`의 REST API와 `mcp_entry.py`의 MCP 기능이 모두 활성화된 통합 서버가 실행됩니다.

- **구현 파일**:

  - `mcp_entry.py`: MCP 관련 라우터(`APIRouter`)와 FastMCP 설정, 에이전트 연동 로직을 작성합니다.
  - `main.py`: `mcp_entry.py`에서 생성한 라우터를 가져와 메인 앱에 등록하는 코드를 추가합니다.

- **의의**:
  - 이 구조는 기능별로 코드를 다른 파일에 분리하여 프로젝트의 가독성과 유지보수성을 높이는 동시에, 단일 진입점을 통해 전체 애플리케이션을 실행하는 FastAPI의 모범적인 프로젝트 구성 방식을 따릅니다.

## 4. 개발 시작점

- 먼저 `requirements.txt` 파일을 생성하고 필요한 라이브러리(`fastapi`, `uvicorn`, `openai`, `python-dotenv` 등)를 명시하는 것부터 시작한다.
- 그 다음, `main.py` 파일을 만들고 기본 FastAPI 앱 구조부터 작성한다.

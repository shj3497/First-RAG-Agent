# AI 에이전트 'MCP 서버' 백엔드 (Python)

이 프로젝트는 사용자의 질문 의도를 파악하고, 사전 정의된 도구(Tools)를 사용하여 최적의 답변을 생성하는 지능형 AI 에이전트 API 서버입니다.

## 🚀 시작하기

### 사전 요구사항

- Python 3.8 이상
- `pip` (Python 패키지 관리자)

### 1. 프로젝트 설정

#### 가상환경 생성 및 활성화

이 프로젝트는 Python 가상환경(`venv`)을 사용하여 의존성을 시스템 전체와 격리하여 관리하는 것을 권장합니다.

1.  **가상환경 생성**
    프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 `venv`라는 이름의 가상환경을 생성합니다.

    ```bash
    python3 -m venv venv
    ```

2.  **가상환경 활성화**
    생성된 가상환경을 활성화합니다.
    ```bash
    source venv/bin/activate
    ```
    터미널 프롬프트 앞에 `(venv)`가 표시되면 성공적으로 활성화된 것입니다.

### 2. 의존성 라이브러리 설치

프로젝트에 필요한 모든 라이브러리는 `requirements.txt` 파일에 명시되어 있습니다.

1.  **Python 라이브러리 설치**
    가상환경이 활성화된 상태에서 아래 명령어를 실행하여 필요한 라이브러리를 모두 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Playwright 브라우저 설치**
    RAG 데이터 수집에 사용되는 헤드리스 브라우저(Playwright)를 설치합니다.
    ```bash
    playwright install
    ```

### 3. 환경변수 설정

OpenAI API 키와 같은 민감한 정보는 `.env` 파일을 통해 관리합니다.

1.  `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.

    ```bash
    cp .env.example .env
    ```

2.  생성된 `.env` 파일을 열어 `"YOUR_OPENAI_API_KEY_HERE"` 부분을 본인의 실제 OpenAI API 키로 교체합니다.
    ```
    OPENAI_API_KEY="sk-..."
    ```

## 🏃‍♀️ 서버 실행

모든 설정이 완료되었다면, 아래 명령어를 실행하여 FastAPI 개발 서버를 시작합니다.

```bash
uvicorn main:app --reload
```

- `--reload` 옵션은 코드가 변경될 때마다 서버를 자동으로 재시작해주는 편리한 기능입니다.

서버가 성공적으로 실행되면 터미널에 `Uvicorn running on http://127.0.0.1:8000` 메시지가 표시됩니다.

## 📖 API 사용 예시

### RAG 데이터베이스 구축

사이트의 `sitemap.xml`을 기반으로 RAG 데이터베이스를 구축합니다. `site_url` 파라미터에 `sitemap.xml`이 위치한 사이트의 URL을 전달합니다.

- **실서버 대상**
  실서버(`https://www.megazone.com`)를 대상으로 RAG를 구축하는 예시입니다.

  ```bash
  curl -X POST "http://127.0.0.1:8000/build-rag" \
  -H "Content-Type: application/json" \
  -d '{"site_url": "https://www.megazone.com"}'
  ```

- **로컬 빌드 결과물 대상**
  Next.js 프로젝트를 로컬에서 빌드하고 실행한 후(`http://localhost:3000`), 해당 로컬 서버를 대상으로 RAG를 구축하는 예시입니다.

  ```bash
  curl -X POST "http://127.0.0.1:8000/build-rag" \
  -H "Content-Type: application/json" \
  -d '{"site_url": "http://localhost:3000"}'
  ```

### AI 에이전트에게 질문하기

서버가 실행 중일 때, 아래 `curl` 명령어를 사용하여 `/query` 엔드포인트로 AI 에이전트에게 질문을 보낼 수 있습니다.

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{"question": "메가존클라우드의 AI 데이터 솔루션에는 어떤 것들이 있나요?"}'
```

## ✅ 데이터베이스 확인

RAG 데이터가 벡터 데이터베이스에 잘 저장되었는지 확인하고 싶을 때, 아래 명령어를 사용할 수 있습니다.

이 스크립트는 DB에 저장된 총 아이템의 개수와 일부 데이터 샘플을 출력합니다.

```bash
python check_vector_db.py
```

### RAG 검색 테스트 (CLI)

API 서버와는 별개로, 터미널에서 직접 RAG 검색 기능의 성능을 테스트해볼 수 있습니다.

```bash
python interactive_rag_test.py
```

스크립트를 실행하면 질문을 입력하라는 안내가 나옵니다. 질문을 입력하면 DB에서 검색된 결과를 바로 확인할 수 있습니다.

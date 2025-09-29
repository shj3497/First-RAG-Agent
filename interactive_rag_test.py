# interactive_rag_test.py
import asyncio
import os
from dotenv import load_dotenv

# core.tools.rag_search 모듈에서 RagSearchTool 클래스를 가져옵니다.
from core.tools.rag_search import RagSearchTool


async def main():
    """
    터미널에서 사용자와 대화하며 RAG 검색 도구를 테스트하는 메인 함수입니다.
    """
    print("--- RAG 검색 도구 대화형 테스트 ---")
    print("API 서버와는 별개로 RAG 검색 기능만 독립적으로 테스트합니다.")
    print("질문을 입력하면 벡터 DB에서 검색한 결과를 보여줍니다.")
    print("------------------------------------")

    # .env 파일에서 OpenAI API 키를 로드합니다.
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 에러: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 API 키를 추가해주세요.")
        return

    # RagSearchTool 인스턴스를 생성합니다.
    rag_tool = RagSearchTool()

    while True:
        try:
            # 사용자로부터 질문을 입력받습니다.
            user_query = input("\n[❓] 질문을 입력하세요 (종료하려면 'exit' 입력): ")

            if user_query.lower() == 'exit':
                print("👋 테스트를 종료합니다.")
                break

            if not user_query.strip():
                continue

            # RAG 도구를 실행하여 검색 결과를 가져옵니다.
            # input() 함수는 비동기가 아니므로, rag_tool.execute()를 await으로 호출합니다.
            search_result = await rag_tool.execute(query=user_query)

            # 검색 결과를 터미널에 출력합니다.
            print("\n--- 📜 검색 결과 ---")
            print(search_result)
            print("--------------------")

        except KeyboardInterrupt:
            # Ctrl+C 를 누르면 종료됩니다.
            print("\n👋 테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때 main 함수를 비동기적으로 실행합니다.
    asyncio.run(main())

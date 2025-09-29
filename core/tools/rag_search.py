# core/tools/rag_search.py
import openai
import chromadb
from typing import Any, List, Dict
from openai import AsyncOpenAI

from core.tools.base import Tool
# rag_builder와 동일한 상수들을 사용하기 위해 import 합니다.
from core.rag_builder import (
    VECTOR_DB_PATH,
    VECTOR_DB_COLLECTION_NAME,
    EMBEDDING_MODEL,
)

# rag_builder의 지연 초기화 헬퍼를 직접 재사용하지 않기 위해, 이 파일에서도 독립적으로 지연 초기화를 구현합니다.
_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    """
    OpenAI 비동기 클라이언트를 지연 생성하여 반환합니다.
    모듈 임포트 시점에 환경변수가 없어도 에러가 발생하지 않도록 합니다.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client

# OpenAI 클라이언트는 지연 초기화합니다.


class RagSearchTool(Tool):
    """
    사용자의 질문을 기반으로 벡터 데이터베이스(ChromaDB)를 검색하여
    관련성 높은 문서 조각(청크)을 찾아 반환하는 도구입니다.
    """

    def __init__(self):
        # 도구가 초기화될 때 ChromaDB 클라이언트와 컬렉션에 연결합니다.
        # 이렇게 하면 execute 메서드가 호출될 때마다 새로 연결할 필요가 없어 효율적입니다.
        self._client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self._collection = self._client.get_collection(
            name=VECTOR_DB_COLLECTION_NAME)

    @property
    def name(self) -> str:
        return "rag_search_tool"

    @property
    def description(self) -> str:
        return (
            "Megazone Cloud(메가존클라우드)의 서비스, 제품, 솔루션, 파트너, 고객사례, "
            "회사 정보, 가이드라인 등 내부 정보에 대한 질문에 답변할 때 사용합니다. "
            "사용자의 질문을 그대로 입력받아 관련 문서를 검색합니다."
        )

    async def execute(self, query: str) -> str:
        """
        주어진 쿼리(질문)를 벡터로 변환하여 DB에서 유사도 높은 문서를 검색하고,
        그 결과를 정리된 문자열로 반환합니다.

        Args:
            query (str): 사용자가 입력한 검색 질문.

        Returns:
            str: 검색된 문서들의 내용을 종합한 최종 문자열.
        """
        print(f"🔍 RAG 검색 실행: '{query}'")

        # 1. 사용자 쿼리를 임베딩 벡터로 변환
        query_embedding = await self._embed_query(query)

        # 2. ChromaDB에서 유사도 검색 실행
        # collection.query는 주어진 벡터와 가장 가까운 아이템들을 찾아줍니다.
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  # 가장 관련성 높은 5개의 결과를 가져옵니다.
            # 메타데이터와 함께 유사도 점수(distance)도 가져옵니다.
            include=["metadatas", "distances"]
        )

        # 3. 검색 결과를 사용자 친화적인 문자열로 포맷팅
        return self._format_results(results, query)

    async def _embed_query(self, query: str) -> List[float]:
        """쿼리 문자열을 OpenAI API를 통해 임베딩 벡터로 변환합니다."""
        response = await _get_openai_client().embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def _format_results(self, results: Dict[str, Any], original_query: str) -> str:
        """ChromaDB 검색 결과를 최종 답변의 근거 자료로 사용하기 좋게 문자열로 만듭니다."""
        if not results or not results.get('ids') or not results['ids'][0]:
            return f"'{original_query}'에 대한 검색 결과가 없습니다."

        formatted_string = f"--- '{original_query}'에 대한 검색 결과 ---\n\n"

        # 검색된 각 문서 조각(청크)의 정보를 순서대로 추가합니다.
        for i, metadata in enumerate(results['metadatas'][0]):
            source_url = metadata.get('page_url', '출처 불명')
            chunk_text = metadata.get('chunk_text', '내용 없음')
            # distance 점수는 낮을수록 유사도가 높다는 의미입니다.
            distance = results['distances'][0][i]

            formatted_string += f"문서 #{i+1}:\n"
            formatted_string += f"  - 출처 URL: {source_url}\n"
            formatted_string += f"  - 유사도 점수: {distance:.4f}\n"
            formatted_string += f"  - 내용:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"

        return formatted_string

    def to_openai_format(self) -> Dict[str, Any]:
        """
        이 도구를 OpenAI의 Tool Calling 명세에 맞게 JSON으로 변환합니다.
        LLM에게 'query'라는 이름의 문자열 파라미터가 필요하다고 알려줍니다.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "벡터 데이터베이스에서 검색할 사용자 질문 또는 검색어",
                        }
                    },
                    "required": ["query"],
                },
            },
        }


# 이 파일이 직접 실행될 때를 위한 간단한 테스트 코드 (예시)
if __name__ == '__main__':
    import asyncio

    async def main():
        rag_tool = RagSearchTool()
        test_query = "메가존클라우드의 AI 데이터 솔루션에 대해 알려줘"
        search_result = await rag_tool.execute(query=test_query)
        print(search_result)

    asyncio.run(main())

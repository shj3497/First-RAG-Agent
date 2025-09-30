# core/tools/rag_search.py
import openai
import chromadb
import pickle
import os
from typing import Any, List, Dict, Tuple
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt  # Mecab 대신 Okt를 임포트

from core.tools.base import Tool
# rag_builder와 동일한 상수들을 사용하기 위해 import 합니다.
from core.rag_builder import (
    VECTOR_DB_PATH,
    VECTOR_DB_COLLECTION_NAME,
    EMBEDDING_MODEL,
    BM25_INDEX_PATH,  # BM25 인덱스 경로 추가
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
    하이브리드 검색(벡터 + 키워드)을 수행하여 관련성 높은 문서 조각(청크)을 찾아 반환하는 도구입니다.
    """

    def __init__(self):
        # 1. 벡터 DB 클라이언트 초기화
        self._client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self._collection = self._client.get_collection(
            name=VECTOR_DB_COLLECTION_NAME)

        # 2. Okt 형태소 분석기 초기화
        self._okt = Okt()
        print("✅ Okt 형태소 분석기를 성공적으로 로드했습니다.")

        # 3. BM25 키워드 인덱스 로드
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, "rb") as f:
                bm25_data = pickle.load(f)
            self._bm25_index: BM25Okapi = bm25_data['bm25_index']
            self._bm25_corpus_ids: List[str] = bm25_data['corpus_ids']
            self._bm25_corpus_chunks: List[str] = bm25_data['corpus_chunks']
            print(
                f"✅ BM25 인덱스를 성공적으로 로드했습니다. ({len(self._bm25_corpus_chunks)}개 문서)")
        else:
            self._bm25_index = None
            print("⚠️ 경고: BM25 인덱스 파일이 없어 키워드 검색을 사용할 수 없습니다.")

    @property
    def name(self) -> str:
        return "hybrid_rag_search_tool"

    @property
    def description(self) -> str:
        return (
            "Megazone Cloud(메가존클라우드)의 서비스, 제품, 솔루션 등 내부 정보에 대한 질문에 답변할 때 사용합니다. "
            "의미가 중요한 질문과 특정 키워드(예: 제품명, 에러코드)가 중요한 질문 모두에 효과적입니다."
        )

    async def execute(self, query: str, n_results: int = 5) -> Tuple[str, int]:
        """
        주어진 쿼리로 벡터 검색과 키워드 검색을 동시에 수행하고,
        결과를 융합(RRF)하여 최종적으로 가장 관련성 높은 문서를 반환합니다.

        Returns:
            Tuple[str, int]: (포맷팅된 최종 결과 문자열, 최종 문서 개수)
        """
        print(f"🔍 하이브리드 검색 실행: '{query}'")

        # 1. 벡터 검색과 키워드 검색을 병렬로 실행
        # 후보군을 더 많이 확보
        vector_results = await self._vector_search(query, n_results * 2)
        keyword_results = self._keyword_search(query, n_results * 2)

        # --- 상세 로그 추가 ---
        print(f"  - ➡️  벡터 검색 결과: {len(vector_results)}개")
        if vector_results:
            print(
                f"    - Top vector result: ID={vector_results[0][0]}, Score={vector_results[0][1]:.4f}")

        print(f"  - ➡️  키워드 검색(BM25) 결과: {len(keyword_results)}개")
        if keyword_results:
            print(
                f"    - Top keyword result: ID={keyword_results[0][0]}, Score={keyword_results[0][1]:.4f}")
        # --- 로그 추가 끝 ---

        # 2. 두 검색 결과를 RRF(Reciprocal Rank Fusion)로 융합
        fused_results = self._reciprocal_rank_fusion(
            [vector_results, keyword_results])

        # --- 상세 로그 추가 ---
        print(f"  - 융합(RRF) 후 최종 후보: {len(fused_results)}개")
        if fused_results:
            print(
                f"    - Top fused result: ID={fused_results[0][0]}, Score={fused_results[0][1]:.4f}")
        # --- 로그 추가 끝 ---

        # 3. 융합된 결과에서 최종 n_results 개수만큼 선택
        # RRF 결과는 (doc_id, score) 튜플의 리스트이므로, doc_id만 추출
        final_doc_ids = [doc_id for doc_id, _ in fused_results[:n_results]]

        # 4. 최종 ID 목록에 해당하는 문서 정보(메타데이터)를 DB에서 가져옴
        if not final_doc_ids:
            return f"'{query}'에 대한 검색 결과가 없습니다.", 0

        final_documents = self._collection.get(
            ids=final_doc_ids, include=["metadatas"])

        # 5. 검색 결과를 LLM이 이해하기 좋은 형태로 포맷팅
        formatted_string = self._format_results(final_documents, query)

        return formatted_string, len(final_doc_ids)

    def _reciprocal_rank_fusion(self, result_sets: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
        """
        여러 검색 결과 목록을 RRF 알고리즘으로 융합하여 재순위화합니다.
        k: 순위 가중치 계산에 사용되는 상수.
        """
        scores: Dict[str, float] = {}
        for results in result_sets:
            for rank, (doc_id, _) in enumerate(results, 1):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank)

        # 점수가 높은 순으로 정렬하여 (doc_id, score) 튜플 리스트 반환
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    async def _vector_search(self, query: str, n_results: int) -> List[Tuple[str, float]]:
        """벡터 DB에서 유사도 검색을 수행하고 (doc_id, score) 리스트를 반환합니다."""
        if not query:
            return []
        query_embedding = await self._embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances"]
        )
        # ChromaDB의 distance는 '유사하지 않은 정도'이므로, 1에서 빼서 점수로 변환 (높을수록 좋음)
        return list(zip(results['ids'][0], [1 - dist for dist in results['distances'][0]]))

    def _keyword_search(self, query: str, n_results: int) -> List[Tuple[str, float]]:
        """BM25 인덱스에서 키워드 검색을 수행하고 (doc_id, score) 리스트를 반환합니다."""
        if self._bm25_index is None:
            return []

        # --- 쿼리 토큰화 방식 변경 ---
        # 사용자의 질문도 Okt를 사용하여 동일하게 명사만 추출합니다.
        tokenized_query = self._okt.nouns(query)
        doc_scores = self._bm25_index.get_scores(tokenized_query)

        # 점수가 높은 순으로 정렬하여 상위 n_results개 만큼 가져옴
        top_n_indices = sorted(
            range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n_results]

        # (doc_id, score) 형태로 변환하여 반환
        return [(self._bm25_corpus_ids[i], doc_scores[i]) for i in top_n_indices]

    async def _embed_query(self, query: str) -> List[float]:
        """쿼리 문자열을 OpenAI API를 통해 임베딩 벡터로 변환합니다."""
        response = await _get_openai_client().embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def _format_results(self, final_documents: Dict[str, Any], original_query: str) -> str:
        """ChromaDB 검색 결과를 최종 답변의 근거 자료로 사용하기 좋게 문자열로 만듭니다."""
        if not final_documents or not final_documents.get('ids'):
            return f"'{original_query}'에 대한 검색 결과가 없습니다."

        formatted_string = f"--- '{original_query}'에 대한 하이브리드 검색 결과 ---\n\n"

        for i, (doc_id, metadata) in enumerate(zip(final_documents['ids'], final_documents['metadatas'])):
            source_url = metadata.get('page_url', '출처 불명')
            chunk_text = metadata.get('chunk_text', '내용 없음')

            formatted_string += f"문서 #{i+1} (ID: {doc_id}):\n"
            formatted_string += f"  - 출처 URL: {source_url}\n"
            formatted_string += f"  - 내용:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"

        return formatted_string

    # to_openai_format 함수는 name과 description만 수정합니다.
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
        search_result, count = await rag_tool.execute(query=test_query)
        print(search_result)
        print(f"최종 문서 개수: {count}")

    asyncio.run(main())

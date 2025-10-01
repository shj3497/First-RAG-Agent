# core/rag_builder.py
import os
import uuid
import hashlib
from typing import List, Dict, Any
import pickle

# --- 새로운 임포트 ---
import chromadb
import tiktoken
import openai
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt  # Mecab 대신 Okt를 임포트

# --- 새로운 임포트 ---
# 헤드리스 브라우저 스크래핑 관련 함수들을 가져옵니다.
# get_page_html 대신 scrape_dynamic_content를 사용합니다.
from core.scraper import get_all_page_urls, scrape_dynamic_content

# --- 상수 정의 ---
# 이 파일(rag_builder.py)의 실제 위치를 기준으로 프로젝트 루트 디렉토리의 절대 경로를 계산합니다.
# 이렇게 하면 스크립트가 어떤 환경에서 실행되더라도 항상 프로젝트 폴더를 정확히 가리킬 수 있습니다.
# os.path.realpath(__file__): 현재 파일의 실제 절대 경로 (.../core/rag_builder.py)
# os.path.dirname(...): 디렉토리 경로만 추출 (.../core)
# os.path.dirname(...): 한 단계 상위 디렉토리로 이동 (...) -> 최종적으로 프로젝트 루트 경로
PROJECT_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT_PATH, "vector_db")

# VECTOR_DB_PATH 디렉토리가 존재하지 않으면 생성합니다. (최초 실행 시 필요할 수 있음)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

VECTOR_DB_COLLECTION_NAME = "mcp_rag_collection"
# BM25 인덱스 파일 경로를 상수로 추가합니다.
BM25_INDEX_PATH = os.path.join(VECTOR_DB_PATH, "bm25_index.pkl")

# OpenAI 임베딩 모델과 Tiktoken 인코더 이름을 상수로 관리합니다.
EMBEDDING_MODEL = "text-embedding-3-small"
ENCODING_NAME = "cl100k_base"
# 텍스트를 나눌 청크의 크기와 청크 간의 겹치는 토큰 수를 정의합니다.
# 겹치는 부분을 두는 이유는 문맥이 잘리는 것을 방지하기 위함입니다.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- 클라이언트 및 인코더 초기화 ---
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_or_create_collection(name=VECTOR_DB_COLLECTION_NAME)
# Tiktoken 인코더를 초기화합니다.
encoding = tiktoken.get_encoding(ENCODING_NAME)
# OpenAI 비동기 클라이언트는 지연 초기화합니다.
# 임포트 시점에 환경변수가 없더라도 에러가 나지 않도록, 실제 사용 시 생성합니다.
_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    """
    OpenAI 비동기 클라이언트를 지연 생성하여 반환합니다.
    환경변수 `OPENAI_API_KEY`가 설정되어 있으면 자동으로 사용합니다.
    """
    global _openai_client
    if _openai_client is None:
        # api_key를 명시 전달하지 않으면 SDK가 OPENAI_API_KEY 환경변수를 읽습니다.
        _openai_client = AsyncOpenAI()
    return _openai_client


# build_rag_from_path 함수를 async로 변경해야 합니다.
# Playwright가 비동기(async) 기반으로 동작하기 때문입니다.
async def build_rag_from_path(site_url: str) -> Dict[str, Any]:
    """
    지정된 URL의 웹사이트를 크롤링하여 RAG 인덱스를 구축하고 업데이트하는 메인 함수입니다.
    헤드리스 브라우저를 사용하여 Suspense 등 동적 콘텐츠를 포함한 최종 HTML을 가져옵니다.

    Args:
        site_url (str): sitemap.xml이 위치한 사이트의 URL (예: https://www.megazone.com).

    Returns:
        Dict[str, Any]: 처리 결과를 담은 딕셔너리.
    """

    # 1. 사이트를 크롤링하여 모든 페이지의 URL 목록을 가져옵니다.
    # 이 함수가 기존의 `_find_html_files`를 대체합니다.
    all_page_urls = await get_all_page_urls(site_url)
    if not all_page_urls:
        return {"status": "error", "message": f"지정된 URL '{site_url}'에서 페이지를 찾을 수 없습니다."}

    # --- 1. 삭제된 페이지 처리 ---
    # DB에 저장된 모든 페이지의 URL을 가져옵니다.
    db_pages = _get_all_pages_from_db()
    crawled_pages_set = set(all_page_urls)

    # DB에는 있지만 최신 크롤링 결과에는 없는 페이지를 찾습니다.
    pages_to_delete = [url for url in db_pages if url not in crawled_pages_set]

    if pages_to_delete:
        print(
            f"삭제된 페이지 {len(pages_to_delete)}개를 DB에서 제거합니다: {pages_to_delete}")
        _delete_vectors_by_url(pages_to_delete)

    # --- 2. 신규/변경된 페이지 처리 ---
    processed_files_count = 0
    skipped_files_count = 0
    for page_url in all_page_urls:
        # get_page_html과 _extract_text_from_html을 합친 새 함수를 호출합니다.
        text_content = await scrape_dynamic_content(page_url)
        if not text_content.strip():
            print(f"Skipped (no content): {page_url}")
            continue

        content_hash = _generate_content_hash(text_content)

        # --- 2a. 콘텐츠 변경 감지 ---
        # DB에서 해당 URL의 기존 메타데이터를 가져옵니다.
        existing_item = collection.get(where={"page_url": page_url}, limit=1)

        # 기존 데이터가 있고, 콘텐츠 해시가 동일하면 변경이 없는 것이므로 건너뜁니다.
        if existing_item['metadatas'] and existing_item['metadatas'][0]['content_hash'] == content_hash:
            print(f"Skipped (no changes): {page_url}")
            skipped_files_count += 1
            continue

        print(f"Processing (new or updated): {page_url}")

        # --- 2b. 신규/변경된 콘텐츠 처리 (기존 로직) ---
        chunks = _split_text_into_chunks(text_content)
        metadatas = [{"page_url": page_url, "content_hash": content_hash,
                      "chunk_text": chunk} for chunk in chunks]
        embeddings = await _embed_chunks(chunks)
        ids = [f"{page_url}#{i}" for i in range(len(chunks))]

        # 기존에 있던 페이지가 업데이트된 경우, 이전 청크들을 먼저 삭제하여 개수 변경에 대응합니다.
        if existing_item['ids']:
            _delete_vectors_by_url([page_url])

        _upsert_vectors_to_db(
            ids=ids, embeddings=embeddings, metadatas=metadatas)

        processed_files_count += 1

    # --- 3. 모든 DB 작업 완료 후 BM25 인덱스 재생성 ---
    _build_and_save_bm25_index()

    return {
        "status": "success",
        "message": f"RAG 인덱스 빌드 완료. 처리: {processed_files_count}개, 변경 없음: {skipped_files_count}개, 삭제: {len(pages_to_delete)}개.",
    }


def _build_and_save_bm25_index():
    """
    ChromaDB에 저장된 모든 문서를 기반으로 BM25 키워드 검색 인덱스를 생성하고 파일로 저장합니다.
    RAG 빌드 프로세스의 마지막에 호출되어 항상 최신 상태를 유지하도록 합니다.
    """

    print("🔄 BM25 인덱스를 재생성합니다...")
    # .get()의 include 파라미터에서 'ids'를 제거합니다. ids는 기본적으로 반환됩니다.
    all_items = collection.get(include=['metadatas'])
    if not all_items or not all_items['ids']:
        print("⚠️ DB에 데이터가 없어 BM25 인덱스를 생성할 수 없습니다.")
        return

    # --- 안정성 강화를 위한 필터링 로직 추가 ---
    # BM25 인덱싱을 위해 ID와 청크 텍스트를 추출하되, 내용이 없는 청크는 제외합니다.
    valid_ids = []
    valid_chunks = []
    for i, metadata in enumerate(all_items['metadatas']):
        chunk_text = metadata.get('chunk_text', '').strip()
        if chunk_text:  # 텍스트가 비어있지 않은 경우에만 추가
            valid_ids.append(all_items['ids'][i])
            valid_chunks.append(chunk_text)

    if not valid_chunks:
        print("⚠️ 유효한 내용이 있는 문서가 없어 BM25 인덱스를 생성할 수 없습니다.")
        return

    # --- 토큰화 방식 변경 ---
    # Okt 형태소 분석기를 사용하여 코퍼스를 토큰화합니다.
    print("Okt 형태소 분석기를 사용하여 토큰화를 시작합니다...")
    okt = Okt()
    # morphs 대신 nouns를 사용하여 명사만 추출, 검색 성능 향상
    tokenized_corpus = [okt.nouns(chunk) for chunk in valid_chunks]
    print("토큰화 완료.")

    # BM25 인덱스 생성
    bm25 = BM25Okapi(tokenized_corpus)

    # 검색 시 원본 청크를 조회하기 위해 인덱스와 원본 데이터를 함께 저장합니다.
    bm25_data = {
        "bm25_index": bm25,
        "corpus_ids": valid_ids,       # 필터링된 ID 리스트 사용
        "corpus_chunks": valid_chunks  # 필터링된 청크 리스트 사용
    }

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"✅ BM25 인덱스 생성 완료. {len(valid_chunks)}개 문서 처리. ({BM25_INDEX_PATH})")


def _generate_content_hash(content: str) -> str:
    """
    주어진 문자열 내용의 SHA256 해시 값을 계산하여 반환합니다.
    이 해시는 파일 내용이 변경되었는지 여부를 판단하는 '지문' 역할을 합니다.
    """
    # hashlib.sha256: 안전한 해시 알고리즘 중 하나입니다.
    # .encode('utf-8'): 해시 계산을 위해 문자열을 바이트로 변환합니다.
    # .hexdigest(): 계산된 해시를 16진수 문자열로 변환합니다.
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _extract_text_from_html(html_content: str) -> str:
    """
    BeautifulSoup4를 사용하여 HTML 콘텐츠에서 불필요한 태그를 제거하고
    본문 텍스트만 추출하여 반환합니다.
    [주의] 이 함수는 동적 탭 콘텐츠 처리를 위해 더 이상 메인 로직에서 사용되지 않습니다.
    대신 scraper.py의 scrape_dynamic_content 함수를 사용해야 합니다.
    """
    soup = BeautifulSoup(html_content, 'lxml')

    # script, style, nav, footer, header 태그와 그 안의 내용을 모두 제거합니다.
    # RAG의 정확도를 높이기 위해 본문과 관련 없는 부분을 제거하는 과정입니다.
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # .get_text() 메서드는 태그들을 제외한 순수 텍스트만 추출합니다.
    # separator=" ": 텍스트 조각들 사이에 공백을 넣어 단어가 붙는 것을 방지합니다.
    # strip=True: 각 텍스트 라인의 앞뒤 공백을 제거합니다.
    text = soup.get_text(separator=" ", strip=True)
    return text


def _split_text_into_chunks(text: str) -> List[str]:
    """
    Tiktoken 인코더를 사용하여 주어진 텍스트를 설정된 크기의 청크로 분할합니다.
    """
    # 텍스트를 토큰 ID의 리스트로 변환합니다.
    tokens = encoding.encode(text)

    chunks = []
    # 전체 토큰 길이를 순회하면서 청크를 생성합니다.
    # CHUNK_SIZE - CHUNK_OVERLAP 만큼 건너뛰면서 겹치는 청크를 만듭니다.
    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        # 토큰 ID 리스트를 다시 문자열로 디코딩하여 청크 리스트에 추가합니다.
        chunks.append(encoding.decode(chunk_tokens))

    return chunks


async def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    OpenAI의 임베딩 API를 사용하여 텍스트 청크 리스트를 벡터 리스트로 변환합니다.
    """
    # OpenAI API는 여러 개의 입력을 한 번에 처리할 수 있어 효율적입니다.
    response = await _get_openai_client().embeddings.create(
        input=chunks,
        model=EMBEDDING_MODEL
    )
    # API 응답에서 실제 임베딩 벡터 데이터만 추출하여 반환합니다.
    return [embedding.embedding for embedding in response.data]


def _upsert_vectors_to_db(ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
    """
    ChromaDB 컬렉션에 ID, 임베딩 벡터, 메타데이터를 저장(upsert)합니다.
    'upsert'는 ID가 존재하면 업데이트하고, 존재하지 않으면 새로 삽입하는 편리한 기능입니다.
    """
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

# TODO: 아래 함수들은 추후 구현이 필요합니다.
# def _split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
#     """Tiktoken 또는 다른 방식을 사용하여 텍스트를 의미있는 조각(청크)으로 나눕니다."""
#     pass

# def _embed_chunks(chunks: List[str]) -> List[List[float]]:
#     """OpenAI의 임베딩 API를 사용하여 텍스트 청크들을 벡터로 변환합니다."""
#     pass

# def _upsert_vectors_to_db(vectors: List[List[float]], metadatas: List[Dict]):
#     """ChromaDB에 벡터와 메타데이터를 저장(upsert)합니다."""
#     pass


def _get_all_pages_from_db() -> List[str]:
    """
    ChromaDB에서 저장된 모든 아이템의 메타데이터를 가져와,
    중복을 제거한 페이지 URL 목록을 반환합니다.
    """
    # .get()에 아무 조건 없이 호출하면 모든 데이터를 가져옵니다.
    # include=['metadatas']로 메타데이터만 효율적으로 가져옵니다.
    all_items = collection.get(include=['metadatas'])

    # 메타데이터에서 'page_url'만 추출하여 set으로 중복을 제거한 후 리스트로 반환합니다.
    if 'metadatas' in all_items and all_items['metadatas']:
        return list(set(item['page_url'] for item in all_items['metadatas']))
    return []


def _delete_vectors_by_url(urls: List[str]):
    """
    주어진 URL 리스트에 해당하는 모든 벡터 데이터를 DB에서 삭제합니다.
    """
    # .delete()의 where 필터를 사용하여 특정 메타데이터 값을 가진 아이템을 삭제할 수 있습니다.
    collection.delete(where={"page_url": {"$in": urls}})

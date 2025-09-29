# check_vector_db.py
import chromadb
from pprint import pprint

# --- 상수 정의 ---
# core/rag_builder.py 와 동일한 경로와 컬렉션 이름을 사용해야 합니다.
VECTOR_DB_PATH = "./vector_db"
VECTOR_DB_COLLECTION_NAME = "mcp_rag_collection"


def check_database_content():
    """
    ChromaDB에 연결하여 저장된 데이터를 확인하고 요약 정보를 출력하는 스크립트입니다.
    """
    print(f"Connecting to ChromaDB at: {VECTOR_DB_PATH}")

    try:
        # DB 클라이언트와 컬렉션에 연결합니다.
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=VECTOR_DB_COLLECTION_NAME)

        # collection.count() 로 컬렉션에 저장된 총 아이템(벡터)의 개수를 가져옵니다.
        total_items = collection.count()

        if total_items == 0:
            print("❌ 데이터베이스에 저장된 아이템이 없습니다.")
            return

        print(f"✅ 데이터베이스 연결 성공!")
        print(f"   - 컬렉션 이름: '{VECTOR_DB_COLLECTION_NAME}'")
        print(f"   - 총 아이템 개수: {total_items}개")

        # collection.get() 으로 실제 데이터를 가져옵니다.
        # include=["metadatas"] 로 메타데이터만 가져와서 내용을 확인합니다.
        # (임베딩 벡터는 너무 길어서 제외)
        retrieved_data = collection.get(
            limit=5,  # 너무 많으면 터미널이 복잡해지니 5개만 가져옵니다.
            include=["metadatas"]
        )

        print("\n--- 저장된 데이터 샘플 (최대 5개) ---")
        # pprint는 딕셔너리나 리스트를 예쁘게 출력해주는 라이브러리입니다.
        pprint(retrieved_data['metadatas'])

    except Exception as e:
        print(f"❌ 데이터베이스 확인 중 오류 발생: {e}")
        print("   - './vector_db' 디렉토리가 존재하는지 확인해주세요.")
        print(f"   - '{VECTOR_DB_COLLECTION_NAME}' 컬렉션이 존재하는지 확인해주세요.")


if __name__ == "__main__":
    check_database_content()

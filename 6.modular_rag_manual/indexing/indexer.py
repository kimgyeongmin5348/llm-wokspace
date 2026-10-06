# indexing/indexer.py
# 전처리 Markdown → medium chunk → Qdrant 색인 파이프라인
# PS E:\langgraph_modular_rag\6.modular_rag_manual> python indexing/indexer.py

import sys
from pathlib import Path

# 6.modular_rag_manual/ 를 sys.path에 추가 → 'common.*', 'indexing.*' 절대 import 가능
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from common.config import (
    CHUNK_MD_PATH,
    QDRANT_URL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_embeddings,
)
from common.qdrant_connect import get_qdrant_client, collection_exists
from indexing.document_loader import load_markdown_page_chunks
from indexing.chunking import make_medium_chunks


# True : collection을 삭제 후 재생성 (첫 실행 또는 데이터 갱신 시)
# False: 이미 저장된 collection을 재사용 (임베딩 비용 없음)
# RECREATE_COLLECTION = False
RECREATE_COLLECTION = True


def run_indexing(
    md_path: Path = CHUNK_MD_PATH,
    recreate: bool = RECREATE_COLLECTION,
) -> QdrantVectorStore:
    """전처리 Markdown page chunk 파일을 읽어 Qdrant에 색인한다."""

    client = get_qdrant_client()
    embeddings = get_embeddings()

    exists = collection_exists(COLLECTION_NAME)
    print(f"collection '{COLLECTION_NAME}' exists: {exists}")

    if recreate:
        print("collection을 새로 생성합니다.")

        # Markdown 파싱 → page-level Document 생성
        pages = load_markdown_page_chunks(md_path)
        medium_chunks = make_medium_chunks(pages)

        print(f"page 수: {len(pages)}, medium chunk 수: {len(medium_chunks)}")

        # force_recreate=True: 동일 이름 collection이 있으면 삭제 후 재생성
        vector_store = QdrantVectorStore.from_documents(
            documents=medium_chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=True,
        )

        print("색인 완료")

    else:
        if not exists:
            raise ValueError(
                f"'{COLLECTION_NAME}' collection이 아직 없습니다. "
                "처음 실행할 때는 RECREATE_COLLECTION = True로 설정하세요."
            )

        # 임베딩 재생성 없이 기존 collection에 연결만 함
        print("기존 collection을 재사용합니다.")

        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
        )

    return vector_store


if __name__ == "__main__":
    vector_store = run_indexing()

    client = get_qdrant_client()
    collection_info = client.get_collection(COLLECTION_NAME)

    print("\n[collection 정보]")
    print("collection name:", COLLECTION_NAME)
    print("points count:", collection_info.points_count)
    print("status:", collection_info.status)

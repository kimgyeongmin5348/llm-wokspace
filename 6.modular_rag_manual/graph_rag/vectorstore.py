from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from common.config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# from_existing_collection: 이미 생성된 collection에 연결만 함 (데이터 삽입 없음)
# from_documents와 달리 서버 시작마다 임베딩을 재생성하지 않아 비용·시간 절약
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
)
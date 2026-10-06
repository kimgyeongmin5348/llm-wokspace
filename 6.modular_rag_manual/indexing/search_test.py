# indexing/search_test.py : 질문으로 Qdrant 검색해 보기
import sys
from pathlib import Path

# Windows 콘솔 출력 인코딩 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_qdrant import QdrantVectorStore
from common.config import COLLECTION_NAME, QDRANT_URL, get_embeddings

question = sys.argv[1] if len(sys.argv) > 1 else "대면 민원 중 폭행이 발생하면 어떻게 해야 하나요?"

# 이미 저장된 서랍(collection)에 연결
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=get_embeddings(),
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
)

print("질문:", question)
for rank, (doc, score) in enumerate(vector_store.similarity_search_with_score(question, k=3), start=1):
    m = doc.metadata
    print(f"\n[{rank}] 유사도 {score:.3f} | p.{m.get('page', '?')} | {m.get('section', '')} / {m.get('topic', '')}")
    print(doc.page_content[:150].replace("\n", " "))

# indexing/chunking.py

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import COLLECTION_NAME, EMBEDDING_MODEL
from indexing.document_loader import load_markdown_page_chunks

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
CHUNK_STRATEGY = "medium"


def make_medium_chunks(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """전처리 Markdown 문서에 적합한 medium chunk를 생성한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Markdown 제목·목록을 우선 분리 기준으로 사용 (표 구조 보존)
        separators=["\n## ", "\n# ", "\n- ", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_id": i,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_strategy": CHUNK_STRATEGY,
                "embedding_model": EMBEDDING_MODEL,
                "collection_name": COLLECTION_NAME,
            }
        )

    return chunks


if __name__ == "__main__":
    from common.config import CHUNK_MD_PATH

    pages = load_markdown_page_chunks(CHUNK_MD_PATH)
    medium_chunks = make_medium_chunks(pages)

    print("페이지 수:", len(pages))
    print("medium chunk 수:", len(medium_chunks))

    if medium_chunks:
        print("\n[첫 번째 chunk metadata]")
        print(medium_chunks[0].metadata)

        print("\n[첫 번째 chunk preview]")
        print(medium_chunks[0].page_content[:500])

# indexing/document_loader.py
# 전처리된 Markdown page chunk 파일을 Document 리스트로 로드한다.
# PS E:\langgraph_modular_rag\6.modular_rag_manual> python indexing/document_loader.py

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.documents import Document


def load_markdown_page_chunks(md_path: Path) -> list[Document]:
    """전처리된 RAG page chunk Markdown 파일을 Document 리스트로 읽는다.

    입력 파일 형식 (노트북 1-2에서 생성):
        ## Page 1 | 섹션명 | 주제명
        <!-- source: ... -->
        본문 내용...

    분리 기준: '## Page n | section | topic' 헤더
    """
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown 파일을 찾을 수 없습니다: {md_path}")

    text = md_path.read_text(encoding="utf-8")

    # 문서 전체 제목(# 제목) 제거
    text = re.sub(r"^# .+?\n", "", text, count=1)

    # '## Page n | section | topic' 앞에서 분리
    parts = re.split(r"(?=^## Page\s+\d+\s*\|)", text, flags=re.MULTILINE)

    docs = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        lines = part.splitlines()
        header = lines[0].strip()

        match = re.match(
            r"^## Page\s+(\d+)\s*\|\s*([^|]+)\s*\|\s*(.+)$",
            header,
        )

        if not match:
            continue

        page = int(match.group(1))
        section = match.group(2).strip()
        topic = match.group(3).strip()

        # HTML comment 메타데이터 라인 제거 (<!-- ... -->)
        content_lines = [
            line for line in lines[1:]
            if not line.strip().startswith("<!--")
        ]

        content = "\n".join(content_lines).strip()

        if not content:
            continue

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": md_path.name,
                    "page": page,
                    "section": section,
                    "topic": topic,
                    "document_type": "preprocessed_markdown",
                    "chunk_type": "page",
                },
            )
        )

    return docs


if __name__ == "__main__":
    from common.config import CHUNK_MD_PATH

    pages = load_markdown_page_chunks(CHUNK_MD_PATH)

    print("page 수:", len(pages))

    if pages:
        print("\n[첫 번째 페이지 metadata]")
        print(pages[0].metadata)

        print("\n[첫 번째 페이지 preview]")
        print(pages[0].page_content[:500])

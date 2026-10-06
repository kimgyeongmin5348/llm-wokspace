import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# 6.modular_rag_manual/ 를 sys.path에 추가해 'common.*' 를 절대 경로로 임포트
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from common.config import (
    QDRANT_URL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    DEFAULT_TOP_K,
)

from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path="../../.env")

# ============================================================
# 1. 요청/응답 모델
# ============================================================

class SearchRequest(BaseModel):
    """검색 요청 모델"""

    query: str = Field(..., min_length=1, description="사용자 질문")
    k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="검색할 문서 개수")


class SearchResult(BaseModel):
    """검색 결과 모델"""

    rank: int
    score: float | None = None
    content: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """검색 응답 모델"""

    query: str
    count: int
    results: list[SearchResult]


class HealthResponse(BaseModel):
    """상태 확인 응답 모델"""

    status: str
    qdrant_url: str
    collection_name: str
    collection_exists: bool
    vectorstore_ready: bool


# ============================================================
# 2. FastAPI lifespan
#    - 서버 시작 시 기존 Qdrant collection에 연결
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI 서비스 시작")
    print("기존 Qdrant collection에 연결합니다.")

    client = QdrantClient(url=QDRANT_URL, timeout=10)

    try:
        # Qdrant 서버 연결 확인
        client.get_collections()

        # collection 존재 여부 확인
        exists = client.collection_exists(
            collection_name=COLLECTION_NAME
        )

        if not exists:
            raise RuntimeError(
                f"'{COLLECTION_NAME}' collection이 존재하지 않습니다. "
                "먼저 오프라인 임베딩 작업을 수행하세요."
            )

        # 임베딩 모델 생성
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # 기존 collection에 연결
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
        )

        # app.state에 저장해 각 엔드포인트에서 전역 객체 없이 접근
        app.state.qdrant_client = client
        app.state.vectorstore = vectorstore
        app.state.collection_exists = exists

        print("Qdrant 연결 성공")
        print(f"collection name: {COLLECTION_NAME}")
        print("vectorstore 준비 완료")

    except Exception as e:
        app.state.qdrant_client = None
        app.state.vectorstore = None
        app.state.collection_exists = False

        print("서비스 시작 중 오류 발생")
        print(str(e))

        raise e

    yield

    print("FastAPI 서비스 종료")


# ============================================================
# 3. FastAPI 앱 생성
# ============================================================

app = FastAPI(
    title="Civil Complaint Manual Search API",
    description="공직자 민원응대 매뉴얼 기반 Qdrant 검색 서비스",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 4. 기본 API
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Civil Complaint Manual Search API is running",
        "docs": "/docs",
        "health": "/health",
        "search": "/search",
    }


# ============================================================
# 5. Health Check API
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    # lifespan에서 초기화 실패 시 app.state에 속성이 없을 수 있어 getattr 기본값 사용
    client = getattr(app.state, "qdrant_client", None)
    vectorstore = getattr(app.state, "vectorstore", None)

    collection_exists = False

    if client is not None:
        try:
            collection_exists = client.collection_exists(
                collection_name=COLLECTION_NAME
            )
        except Exception:
            collection_exists = False

    return HealthResponse(
        status="ok" if vectorstore is not None else "error",
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        collection_exists=collection_exists,
        vectorstore_ready=vectorstore is not None,
    )


# ============================================================
# 6. 문서 검색 API
# ============================================================

@app.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):
    vectorstore = getattr(app.state, "vectorstore", None)

    if vectorstore is None:
        raise HTTPException(
            status_code=500,
            detail="vectorstore가 준비되지 않았습니다. Qdrant 연결 상태를 확인하세요.",
        )

    try:
        # similarity_search_with_score: 코사인 유사도 점수를 함께 반환
        # 클라이언트가 관련도 임계값 필터링에 활용 가능
        docs_with_scores = vectorstore.similarity_search_with_score(
            query=request.query,
            k=request.k,
        )

        results = []

        for rank, (doc, score) in enumerate(docs_with_scores, start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score) if score is not None else None,
                    content=doc.page_content,
                    metadata=doc.metadata,
                )
            )

        return SearchResponse(
            query=request.query,
            count=len(results),
            results=results,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}",
        )
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8090)
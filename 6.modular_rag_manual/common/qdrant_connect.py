# common/qdrant_connect.py

from functools import lru_cache

from qdrant_client import QdrantClient

from common.config import COLLECTION_NAME, QDRANT_URL, QDRANT_TIMEOUT


# maxsize=1: 동일 URL에 대해 커넥션 풀을 재사용하기 위해 클라이언트를 싱글톤으로 유지
@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """QdrantClient를 생성한다."""

    return QdrantClient(
        url=QDRANT_URL,
        timeout=QDRANT_TIMEOUT,
    )


def check_qdrant_connection() -> None:
    """Qdrant Server 연결 상태를 확인한다."""

    client = get_qdrant_client()

    try:
        collections = client.get_collections().collections

        print("Qdrant 연결 성공")
        print("현재 collection 목록:")

        if not collections:
            print("- collection 없음")
            return

        for collection in collections:
            print(f"- {collection.name}")

    except Exception as e:
        # 연결 실패 원인을 원본 예외로 체이닝해 스택 트레이스를 보존
        raise RuntimeError(
            "Qdrant Server에 연결할 수 없습니다. "
            "Docker Desktop과 Qdrant 컨테이너 실행 상태를 확인하세요. "
            "기본 주소: http://localhost:6333/dashboard"
        ) from e


def collection_exists(collection_name: str = COLLECTION_NAME) -> bool:
    """Qdrant collection 존재 여부를 확인한다."""

    client = get_qdrant_client()

    try:
        return client.collection_exists(collection_name)

    except AttributeError:
        # qdrant_client < 1.7 에는 collection_exists()가 없어 목록 조회로 대체
        collection_names = [
            collection.name
            for collection in client.get_collections().collections
        ]
        return collection_name in collection_names


if __name__ == "__main__":
    check_qdrant_connection()

    print("collection:", COLLECTION_NAME)
    print("Qdrant client 준비 완료")
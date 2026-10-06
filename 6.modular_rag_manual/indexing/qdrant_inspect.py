# indexing/qdrant_inspect.py
# PS E:\langgraph_modular_rag\6.modular_rag_manual> python -m indexing.qdrant_inspect

from pprint import pprint

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models

from common.config import COLLECTION_NAME
from common.qdrant_connect import get_qdrant_client


def print_collection_info(
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Qdrant collection 정보를 확인한다."""
    if client is None:
        client = get_qdrant_client()

    collection_info = client.get_collection(collection_name)

    print("\n" + "=" * 80)
    print("[Qdrant collection 정보]")
    print("=" * 80)

    print("collection name:", collection_name)
    print("points count:", collection_info.points_count)

    indexed_vectors_count = getattr(
        collection_info,
        "indexed_vectors_count",
        None,
    )

    if indexed_vectors_count is None:
        indexed_vectors_count = getattr(
            collection_info,
            "vectors_count",
            "확인 불가",
        )

    print("vectors count:", indexed_vectors_count)
    print("status:", collection_info.status)

    print("\n[collection config]")
    pprint(collection_info.config)


def print_sample_payloads(
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    limit: int = 3,
) -> None:
    """Qdrant point payload 일부를 확인한다."""
    if client is None:
        client = get_qdrant_client()

    points, next_page_offset = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    print("\n" + "=" * 80)
    print("[Qdrant payload 샘플]")
    print("=" * 80)

    for i, point in enumerate(points, start=1):
        print("=" * 80)
        print(f"[{i}] point id:", point.id)

        payload = point.payload or {}

        print("payload keys:", list(payload.keys()))

        print("\nmetadata:")
        pprint(payload.get("metadata"))

        print("\npage_content preview:")
        page_content = payload.get("page_content", "")
        print(page_content[:500].replace("\n", " "))

    print("\nnext_page_offset:", next_page_offset)


def run_similarity_search_test(
    vector_store: QdrantVectorStore,
    question: str,
    k: int = 3,
) -> None:
    """similarity_search_with_score 검색 테스트를 수행한다."""
    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    print("\n" + "=" * 80)
    print("[similarity_search_with_score 테스트]")
    print("=" * 80)
    print("question:", question)

    for i, (doc, score) in enumerate(results, start=1):
        print("=" * 80)
        print(f"[{i}] score: {score}")

        print("metadata:")
        pprint(doc.metadata)

        print("\ncontent preview:")
        print(doc.page_content[:700].replace("\n", " "))


def run_filtered_search_test(
    vector_store: QdrantVectorStore,
    question: str,
    k: int = 3,
    chunk_strategy: str = "medium",
) -> None:
    """metadata filter를 적용한 검색 테스트를 수행한다."""
    filtered_results = vector_store.similarity_search_with_score(
        query=question,
        k=k,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.chunk_strategy",
                    match=models.MatchValue(value=chunk_strategy),
                )
            ]
        ),
    )

    print("\n" + "=" * 80)
    print("[metadata filter 검색 테스트]")
    print("=" * 80)
    print("question:", question)
    print("filter: metadata.chunk_strategy =", chunk_strategy)

    for i, (doc, score) in enumerate(filtered_results, start=1):
        print("=" * 80)
        print(f"[{i}] score: {score}")
        print("page:", doc.metadata.get("page"))
        print("chunk_id:", doc.metadata.get("chunk_id"))
        print("chunk_strategy:", doc.metadata.get("chunk_strategy"))
        print(doc.page_content[:500].replace("\n", " "))


def run_retriever_test(
    vector_store: QdrantVectorStore,
    question: str,
    k: int = 3,
) -> None:
    """VectorStore를 Retriever로 변환해 검색 테스트를 수행한다."""
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    retrieved_docs = retriever.invoke(question)

    print("\n" + "=" * 80)
    print("[Retriever 검색 테스트]")
    print("=" * 80)
    print("question:", question)

    for i, doc in enumerate(retrieved_docs, start=1):
        print("=" * 80)
        print(f"[{i}] metadata")
        pprint(doc.metadata)

        print("\ncontent preview")
        print(doc.page_content[:500].replace("\n", " "))


if __name__ == "__main__":
    client = get_qdrant_client()

    print_collection_info(client)
    print_sample_payloads(client, limit=3)
from .graph_state import ModularRAGState
from .classifier import classify_question
from .vectorstore import vector_store
from common.utils import format_docs
from .chains import evidence_chain, answer_chain, validation_chain


def classify_question_node(state: ModularRAGState) -> dict:
    result = classify_question(state["question"])

    return {
        "case_type": result["case_type"],
        "rewritten_query": result["rewritten_query"],
    }


def retrieve_manual_node(state: ModularRAGState) -> dict:
    # rewritten_query가 있으면 확장 쿼리로, 없으면 원본 질문으로 검색
    query = state.get("rewritten_query") or state["question"]
    docs = vector_store.similarity_search(query, k=4)

    # 검색 결과를 프롬프트 삽입용 포맷 문자열로 변환
    context = format_docs(docs)

    return {
        "documents": docs,
        "context": context,
    }


def organize_evidence_node(state: ModularRAGState) -> dict:
    # 방대한 context에서 질문 관련 핵심 근거만 먼저 추려
    # 다음 노드에서 답변 품질을 높이는 2단계 구조 (Retrieve → Refine)
    evidence = evidence_chain.invoke({
        "question": state["question"],
        "context": state["context"],
    })

    return {
        "evidence": evidence,
    }


def generate_answer_node(state: ModularRAGState) -> dict:
    # raw context 대신 evidence를 사용해 환각(hallucination)을 줄임
    answer = answer_chain.invoke({
        "question": state["question"],
        "evidence": state["evidence"],
    })

    return {
        "answer": answer,
    }


def validate_answer_node(state: ModularRAGState) -> dict:
    # 답변이 근거에 기반하는지 독립적으로 검토 (Self-RAG 패턴)
    # validation 결과는 현재 로깅 목적이며, 조건부 엣지로 재검색 트리거에 활용 가능
    validation = validation_chain.invoke({
        "question": state["question"],
        "context": state["context"],
        "answer": state["answer"],
    })

    return {
        "validation": validation,
    }
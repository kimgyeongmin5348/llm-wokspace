import sys
from pathlib import Path

from langgraph.graph import StateGraph, START, END

# 6.modular_rag_manual/ 를 sys.path에 추가해 'common.*', 'graph_rag.*' 를 절대 경로로 임포트
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graph_rag.graph_state import ModularRAGState
from graph_rag.graph_nodes import (
    classify_question_node,
    retrieve_manual_node,
    organize_evidence_node,
    generate_answer_node,
    validate_answer_node,
)


def build_graph():
    graph = StateGraph(ModularRAGState)

    # 노드 등록: 각 노드는 state를 받아 업데이트할 필드만 dict로 반환
    graph.add_node("classify_question", classify_question_node)
    graph.add_node("retrieve_manual", retrieve_manual_node)
    graph.add_node("organize_evidence", organize_evidence_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("validate_answer", validate_answer_node)

    # 선형 파이프라인: 분류 → 검색 → 근거정리 → 답변생성 → 검증
    # 조건부 엣지 없이 모든 노드를 순서대로 실행
    graph.add_edge(START, "classify_question")
    graph.add_edge("classify_question", "retrieve_manual")
    graph.add_edge("retrieve_manual", "organize_evidence")
    graph.add_edge("organize_evidence", "generate_answer")
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_edge("validate_answer", END)

    # compile()이 호출되어야 실제 실행 가능한 CompiledGraph가 반환됨
    return graph.compile()




if __name__ == "__main__":
    
    app = build_graph()

    question = "대면 민원 중 폭행이 발생하면 어떻게 해야 하나요?"
    print(f"질문: {question}\n")
    result = app.invoke({
        "question": question
    })
    print("\n[답변 생성중입니다. 잠시만 기다려주세요...]\n")
    print("──────────────────────────────────────────────\n")
    print("[질문 유형]")
    print(result["case_type"])

    print("\n[검색 질의]")
    print(result["rewritten_query"])

    print("\n[근거 정리]")
    print(result["evidence"])

    print("\n[답변]")
    print(result["answer"])

    print("\n[검증]")
    print(result["validation"])

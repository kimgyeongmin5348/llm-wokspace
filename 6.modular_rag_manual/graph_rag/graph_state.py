from typing_extensions import TypedDict, NotRequired
from langchain_core.documents import Document


class ModularRAGState(TypedDict):
    # 사용자 입력 - 그래프 진입 시 반드시 제공되어야 하는 유일한 필드
    question: str

    # NotRequired: 해당 노드 실행 전까지는 state에 존재하지 않아도 됨
    # classify_question_node가 채움
    case_type: NotRequired[str]          # 분류된 민원 유형 (예: "violence", "phone_abuse")
    rewritten_query: NotRequired[str]    # 키워드 확장된 검색용 쿼리

    # retrieve_manual_node가 채움
    documents: NotRequired[list[Document]]  # 벡터 검색으로 가져온 원본 청크 목록
    context: NotRequired[str]               # 프롬프트에 삽입할 포맷팅된 문자열

    # organize_evidence_node가 채움
    evidence: NotRequired[str]   # context에서 LLM이 추린 핵심 근거

    # generate_answer_node가 채움
    answer: NotRequired[str]     # 최종 사용자 답변

    # validate_answer_node가 채움
    validation: NotRequired[str] # 근거 충족 여부: '충분' / '부분충분' / '부족'
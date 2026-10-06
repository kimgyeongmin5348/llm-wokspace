def classify_question(question: str) -> dict:
    # 규칙 우선순위: 리스트 순서대로 첫 매칭에서 즉시 반환 (early-exit)
    # 더 구체적인 유형을 앞에 배치해 모호한 키워드에 의한 오분류를 방지
    rules = [
        ("repeat_call", ["반복", "계속 전화", "같은 말", "동일"]),
        ("long_call", ["장시간", "오래", "권장시간", "20분", "15분"]),
        ("supervisor_request", ["상급자", "기관장", "부서장", "팀장"]),
        ("phone_abuse", ["전화", "욕설", "폭언", "협박", "성희롱"]),
        ("face_to_face_abuse", ["대면", "방문", "욕설", "폭언"]),
        ("violence", ["폭행", "때리", "물리"]),
        ("property_damage", ["파손", "부수", "집기", "물품"]),
        ("dangerous_object", ["위험물", "흉기", "칼", "인화물질", "불"]),
        ("online_abuse", ["온라인", "문서", "인터넷"]),
        ("recovery_protection", ["보호", "휴식", "휴가", "심리", "치료"]),
        ("legal_basis", ["법", "처벌", "형량", "고소", "고발"]),
    ]

    for case_type, keywords in rules:
        if any(keyword in question for keyword in keywords):
            return {
                "case_type": case_type,
                "rewritten_query": rewrite_query(question, case_type),
            }

    # 어떤 규칙에도 해당하지 않으면 원본 질문을 그대로 검색 쿼리로 사용
    return {
        "case_type": "general",
        "rewritten_query": question,
    }


def rewrite_query(question: str, case_type: str) -> str:
    # 유형별로 벡터 검색 적중률을 높이는 키워드를 원본 질문 뒤에 덧붙임
    # 매뉴얼 문서에 실제로 등장하는 용어 위주로 구성
    expansions = {
        "repeat_call": "정당한 사유 없는 반복전화 동일민원 통화 종료 상담 종료",
        "long_call": "정당한 사유 없는 장시간 통화 권장시간 상담 종료",
        "supervisor_request": "상급자 기관장 통화 요구 부서장 팀장 연결 요청",
        "phone_abuse": "전화민원 욕설 폭언 협박 성희롱 통화 종료 녹음 고지",
        "face_to_face_abuse": "대면민원 방문민원 욕설 폭언 퇴거 요청 안전요원",
        "violence": "폭행 물리적 위협 신체 피해 경찰 신고 증거 확보",
        "property_damage": "기물 파손 물품 파손 집기 파손 피해 복구 증거 확보",
        "dangerous_object": "위험물 흉기 인화물질 반입 제지 대피 경찰 신고",
        "online_abuse": "온라인 문서 인터넷 민원 욕설 모욕 명예훼손",
        "recovery_protection": "담당자 보호 휴식 심리상담 치료 회복 지원",
        "legal_basis": "민원처리법 형법 고소 고발 처벌 법적 근거",
    }

    expansion = expansions.get(case_type, "")
    return f"{question} {expansion}".strip()
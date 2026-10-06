from langchain_core.prompts import ChatPromptTemplate

# 3단계 프롬프트 파이프라인:
#   1) evidence_prompt : context(원본 청크) → 질문 관련 근거만 추출
#   2) answer_prompt   : evidence(정제된 근거) → 구조화된 최종 답변 생성
#   3) validation_prompt: (질문 + context + 답변) → 근거 충족도 평가
# raw context를 바로 답변에 쓰지 않고 evidence 단계를 거쳐 환각을 줄임

evidence_prompt = ChatPromptTemplate.from_messages([
    ("system", """
검색된 문단에서 질문 답변에 필요한 핵심 근거만 정리하세요.
문단에 없는 내용은 추가하지 마세요.
"""),
    ("human", """
질문:
{question}

검색 문단:
{context}
""")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 공직자 민원응대 매뉴얼 기반 업무지원 AI입니다.
반드시 제공된 근거에 기반하여 답변하세요.
매뉴얼에 없는 내용은 추측하지 마세요.

답변 형식:
1. 핵심 대응
2. 단계별 조치
3. 사용할 수 있는 안내 표현
4. 담당자 보호조치
5. 주의사항
"""),
    ("human", """
질문:
{question}

근거:
{evidence}
""")
])

# validation_prompt는 evidence가 아닌 context(원본 청크)를 사용해
# LLM이 근거를 요약하는 과정에서 생긴 편향 없이 원문 기준으로 검증
validation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 RAG 답변 검토자입니다.
답변이 근거에 기반하는지 검토하세요.
결과는 '충분', '부분충분', '부족' 중 하나로 시작하세요.
"""),
    ("human", """
질문:
{question}

검색 근거:
{context}

답변:
{answer}
""")
])
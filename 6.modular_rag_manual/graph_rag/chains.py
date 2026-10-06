from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from common.config import LLM_MODEL
from .prompts import evidence_prompt, answer_prompt, validation_prompt

# temperature=0: 민원응대 답변은 일관성이 중요하므로 무작위성 제거
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# LCEL 파이프라인: prompt → LLM 호출 → 문자열 파싱
# 각 chain은 노드에서 .invoke({"question": ..., "context": ...}) 형태로 호출됨
evidence_chain = evidence_prompt | llm | StrOutputParser()   # context → 핵심 근거 추출
answer_chain = answer_prompt | llm | StrOutputParser()       # evidence → 최종 답변 생성
validation_chain = validation_prompt | llm | StrOutputParser() # answer 근거 충족도 검토
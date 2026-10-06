# LangGraph 기반 Modular RAG

## 실습 목표

`공직자 민원응대 핵심 매뉴얼.pdf`를 활용하여 다음 흐름으로 RAG를 단계적으로 구현한다.

1. 환경셋팅 및 LangChain Basic 1.x
2. Naive RAG
3. Advanced RAG
4. LangGraph Basic
5. Modular RAG 및 코드 모듈화

## 실행 방법

```bash
uv sync
uv run python -m ipykernel install --user --name langgraph_modular_rag --display-name "langgraph_modular_rag"
```

`.env` 파일에 `OPENAI_API_KEY`를 입력한 뒤 실행한다.

```bash
uv run python -m src.main
```

## 프로젝트 구조

- `notebooks/`: 단계별 수업용 Jupyter Notebook
- `src/`: 최종 모듈화 코드
- `data/`: PDF 원본 데이터

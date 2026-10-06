def preview_text(text: str, max_len: int = 200) -> str:
    if text is None:
        return ""

    text = str(text).replace("\n", " ")
    return text[:max_len] + "..." if len(text) > max_len else text


def print_stream_outputs(app, initial_state: dict) -> None:
    print("=== stream() 실행 결과 ===")

    # stream_mode="updates": 노드 실행이 끝날 때마다 해당 노드의 state 변경분만 yield
    # (기본값 "values"는 전체 state를 매번 반환해 중간 결과 확인이 번거로움)
    for event in app.stream(initial_state, stream_mode="updates"):
        # event = {"노드이름": {노드가 반환한 dict}} 구조
        node_name = list(event.keys())[0]
        node_output = event[node_name]

        print(f"\n[{node_name}]")

        if node_name == "classify_question":
            print("case_type:", node_output.get("case_type"))
            print("rewritten_query:", preview_text(node_output.get("rewritten_query")))

        elif node_name == "retrieve_manual":
            documents = node_output.get("documents", [])
            context = node_output.get("context", "")

            print("검색 문서 수:", len(documents))
            print("context 길이:", len(context))

        elif node_name == "organize_evidence":
            print("evidence:", preview_text(node_output.get("evidence")))

        elif node_name == "generate_answer":
            print("answer:", preview_text(node_output.get("answer")))

        elif node_name == "validate_answer":
            print("validation:", preview_text(node_output.get("validation")))
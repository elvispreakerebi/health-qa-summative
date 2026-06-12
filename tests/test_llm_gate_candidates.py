from scripts.llm_gate_candidates import parse_decision


def test_parse_decision_reads_json_choice() -> None:
    decision = parse_decision('{"choice": "B", "confidence": 0.82, "reason": "more direct"}')

    assert decision["choice"] == "B"
    assert decision["confidence"] == 0.82
    assert decision["reason"] == "more direct"


def test_parse_decision_strips_thinking_and_clamps_confidence() -> None:
    decision = parse_decision('<think>notes</think>{"choice": "B", "confidence": 1.8, "reason": "ok"}')

    assert decision["choice"] == "B"
    assert decision["confidence"] == 1.0


def test_parse_decision_fails_closed_to_baseline() -> None:
    decision = parse_decision("I am not sure.")

    assert decision["choice"] == "A"
    assert decision["confidence"] == 0.0

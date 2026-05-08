from __future__ import annotations
from chatree_deka.state import TrialState
from chatree_deka.agents import evaluator_agent


def evaluator_node(state: TrialState) -> dict:
    case_facts = state.get("case_facts", "")
    transcript = state.get("transcript", [])
    judge_ruling = ""

    for turn in reversed(transcript):
        if turn.get("role") == "judge":
            judge_ruling = turn.get("content", "")
            break

    statement = evaluator_agent.act(case_facts, transcript, judge_ruling)

    evaluation_result = "partially_correct"
    lowered = statement.lower()
    if "incorrect" in lowered:
        evaluation_result = "incorrect"
    elif "correct" in lowered:
        evaluation_result = "correct"

    return {
        "current_speaker": "evaluator",
        "evaluation_result": evaluation_result,
        "evaluation_reason": statement,
        "transcript": [{"role": "evaluator", "content": statement, "valid": True}],
    }

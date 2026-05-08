from __future__ import annotations
from chatree_deka.state import TrialState
from chatree_deka.agents import plaintiff_agent


def plaintiff_node(state: TrialState) -> dict:
    mode = state.get("plaintiff_mode", "ai")

    if mode == "manual":
        statement = state.get("pending_statement")
        if not statement:
            statement = "[No statement provided manually]"
        return {"current_speaker": "plaintiff"}

    facts = state.get("case_facts", "")
    transcript = state.get("transcript", [])
    statement = plaintiff_agent.act(facts, transcript)

    return {"current_speaker": "plaintiff", "pending_statement": statement}

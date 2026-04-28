from __future__ import annotations
from chatree_deka.state import TrialState
from chatree_deka.agents import prosecutor_agent

def prosecutor_node(state: TrialState) -> dict:
    mode = state.get("prosecutor_mode", "ai")
    
    if mode == "manual":
        # In manual mode, the user input must be injected via `pending_statement`
        # prior to this node resuming execution, or we use what's already there.
        statement = state.get("pending_statement")
        if not statement:
           statement = "[No statement provided manually]"
        return {"current_speaker": "prosecutor"}
        
    facts = state.get("case_facts", "")
    transcript = state.get("transcript", [])
    statement = prosecutor_agent.act(facts, transcript)
    
    return {"current_speaker": "prosecutor", "pending_statement": statement}

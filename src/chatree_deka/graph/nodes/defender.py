from __future__ import annotations
from chatree_deka.state import TrialState
from chatree_deka.agents import defender_agent

def defender_node(state: TrialState) -> dict:
    mode = state.get("defender_mode", "ai")
    
    if mode == "manual":
        statement = state.get("pending_statement")
        if not statement:
           statement = "[No statement provided manually]"
        return {"current_speaker": "defender"}
        
    facts = state.get("case_facts", "")
    transcript = state.get("transcript", [])
    statement = defender_agent.act(facts, transcript)
    
    return {"current_speaker": "defender", "pending_statement": statement}

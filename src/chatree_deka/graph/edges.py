from __future__ import annotations
from chatree_deka.state import TrialState

def validation_route(state: TrialState) -> str:
    """
    Decides where to route after the validation_node finishes.
    """
    # Validation pass
    if state.get("validation_result") == "pass":
        return "phase_router_after_val"
    
    # Validation fail, can retry
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    speaker = state.get("current_speaker")
    
    if retry_count < max_retries:
        return f"{speaker}_node"
    
    # Validation fail max times, automatic objection by judge
    return "judge_node"

def phase_router_after_val(state: TrialState) -> str:
    """
    Advances to the next turn based on the current phase.
    Note: Phase progression itself should be handled sequentially.
    """
    if state.get("objection_pending"):
        return "judge_node"
        
    phase = state.get("phase")
    speaker = state.get("current_speaker")
    
    if phase == "opening_prosecution" and speaker == "prosecutor":
        return "defender_node"
    elif phase == "opening_defense" and speaker == "defender":
        return "prosecutor_node"
    elif phase == "direct_examination" and speaker == "prosecutor":
        return "defender_node"
    elif phase == "cross_examination" and speaker == "defender":
        return "prosecutor_node"
    elif phase == "closing_prosecution" and speaker == "prosecutor":
        return "defender_node"
    elif phase == "closing_defense" and speaker == "defender":
        return "judge_node"
    
    # If unhandled, hand off to judge
    return "judge_node"

def post_judge_router(state: TrialState) -> str:
    """
    Decides routing after judge speaks.
    """
    phase = state.get("phase")
    if phase == "verdict":
        return "END"
        
    if phase in ["opening_prosecution", "direct_examination", "closing_prosecution"]:
        return "prosecutor_node"
    elif phase in ["opening_defense", "cross_examination", "closing_defense"]:
        return "defender_node"
        
    return "END"

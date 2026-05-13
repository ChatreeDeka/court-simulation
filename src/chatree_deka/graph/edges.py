from __future__ import annotations
from chatree_deka.state import TrialState

def validation_route(state: TrialState) -> str:
    """
    Decides where to route after the validation_node finishes.
    """
    # Validation pass
    if state.get("validation_result") == "pass":
        return "judge_node"
    
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

def judge_router(state: TrialState) -> str:
    """
    Routes based on judge's decision about who should speak next.
    """
    next_speaker = state.get("next_speaker")
    
    if next_speaker == "prosecutor":
        return "prosecutor_node"
    elif next_speaker == "defender":
        return "defender_node"
    elif next_speaker == "judge":
        return "judge_node"
    elif next_speaker == "advance_phase":
        return "advance_phase_node"
    elif next_speaker == "end_trial":
        return "summary_node"
    else:
        # Default fallback
        return "summary_node"

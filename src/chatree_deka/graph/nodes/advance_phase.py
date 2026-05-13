from __future__ import annotations
from chatree_deka.state import TrialState

def advance_phase_node(state: TrialState) -> dict:
    """
    Advances the trial to the next phase.
    """
    current_phase = state.get("phase")
    phases = [
        "opening_prosecution", "opening_defense", 
        "direct_examination", "cross_examination", 
        "closing_prosecution", "closing_defense", "verdict"
    ]
    
    try:
        current_idx = phases.index(current_phase)
        next_idx = current_idx + 1
        if next_idx < len(phases):
            next_phase = phases[next_idx]
        else:
            next_phase = "verdict"
    except ValueError:
        # If phase not found, default to verdict
        next_phase = "verdict"
    
    # Reset retry count for new phase
    return {
        "phase": next_phase,
        "retry_count": 0,
        "next_speaker": None  # Clear the routing decision
    }
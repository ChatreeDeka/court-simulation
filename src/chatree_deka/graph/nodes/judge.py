from __future__ import annotations
from chatree_deka.state import TrialState
from chatree_deka.agents import judge_agent

def judge_node(state: TrialState) -> dict:
    # If the judge is called because of max_retries validation failure,
    # generate an automatic objection rather than calling the LLM/operator
    validation_status = state.get("validation_result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if validation_status == "fail" and retry_count >= max_retries:
        ruling = f"ศาลขอตัดบท เนื่องจาก{state.get('current_speaker')}อ้างอิงผิดพลาดซ้ำหลายครั้ง"
        
        # Advance phase
        current_phase = state.get("phase")
        phases = [
            "opening_prosecution", "opening_defense", 
            "direct_examination", "cross_examination", 
            "closing_prosecution", "closing_defense", "verdict"
        ]
        next_phase_idx = phases.index(current_phase) + 1 if current_phase in phases else len(phases)-1
        next_phase = phases[next_phase_idx] if next_phase_idx < len(phases) else "verdict"

        return {
            "current_speaker": "judge",
            "transcript": [{"role": "judge", "content": ruling, "valid": True}],
            "objection_pending": False,
            "retry_count": 0, # Reset for next phase
            "phase": next_phase
        }

    mode = state.get("judge_mode", "ai")
    
    if mode == "manual":
        statement = state.get("pending_statement")
        if not statement:
           statement = "[No ruling provided manually]"
        return {
            "current_speaker": "judge",
            "transcript": [{"role": "judge", "content": statement, "valid": True}],
            "objection_pending": False,
            "retry_count": 0
        }
        
    facts = state.get("case_facts", "")
    transcript = state.get("transcript", [])
    objection = state.get("objection_pending", False)
    phase = state.get("phase", "")
    statement = judge_agent.act(facts, transcript, objection, phase)
    
    return {
        "current_speaker": "judge", 
        "transcript": [{"role": "judge", "content": statement, "valid": True}],
        "objection_pending": False, # Clear flag
        "retry_count": 0
    }

from __future__ import annotations
from chatree_deka.state import TrialState
from chatree_deka.validation import checker

def validation_node(state: TrialState) -> dict:
    statement = state.get("pending_statement", "")
    speaker = state.get("current_speaker", "unknown")
    
    if speaker == "judge":
        # Judge statements typically don't fail trial validation rules in POC
        # Commit to transcript immediately
        return {
            "validation_result": "pass",
            "transcript": [{"role": speaker, "content": statement, "valid": True}],
            "pending_statement": None
        }

    # Step 1: Pydantic Citation Check
    cite_result = checker.citation_check(statement)
    if not cite_result.valid:
        return {
            "validation_result": "fail",
            "validation_reason": cite_result.reason,
            "retry_count": state.get("retry_count", 0) + 1
        }
        
    # Step 2: LlamaIndex Grounding Check
    ground_result = checker.grounding_check(statement, cite_result.citations_found)
    if not ground_result.valid:
        return {
            "validation_result": "fail",
            "validation_reason": ground_result.reason,
            "retry_count": state.get("retry_count", 0) + 1
        }
        
    # Both passed -> commit to transcript
    return {
        "validation_result": "pass",
        "validation_reason": None,
        "retry_count": 0,
        "transcript": [{"role": speaker, "content": statement, "valid": True}],
        "pending_statement": None
    }

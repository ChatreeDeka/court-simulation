from __future__ import annotations
from typing import Literal, Optional, Annotated
import operator
from typing_extensions import TypedDict

class TrialState(TypedDict):
    # Identity
    case_id: str
    phase: Literal[
        "opening_prosecution",
        "opening_defense",
        "direct_examination",
        "cross_examination",
        "closing_prosecution",
        "closing_defense",
        "verdict",
        "summary"
    ]

    # Role assignment
    judge_mode:      Literal["manual", "ai"]
    prosecutor_mode: Literal["manual", "ai"]
    defender_mode:   Literal["manual", "ai"]

    # Content
    case_facts:        str
    transcript:        Annotated[list[dict], operator.add]   # {"role": str, "content": str, "valid": bool}
    pending_statement: Optional[str]

    # Validation state
    validation_result: Optional[Literal["pass", "fail"]]
    validation_reason: Optional[str]
    retry_count:  int
    max_retries:  int               # Fixed at 2 in POC config

    # Evaluation state
    evaluation_result: Optional[Literal["correct", "partially_correct", "incorrect"]]
    evaluation_reason: Optional[str]

    # Control flow
    objection_pending: bool
    current_speaker:   Literal["judge", "prosecutor", "defender", "evaluator"]
    next_speaker:      Optional[Literal["prosecutor", "defender", "judge", "advance_phase", "end_trial"]]

    # Execution mode
    mode: Literal["run", "train", "coached"]

    # Post-trial outputs
    episode_reward: Optional[float]
    summary:        Optional[dict]

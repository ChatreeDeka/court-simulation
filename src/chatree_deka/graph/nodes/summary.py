from __future__ import annotations
from typing import Optional
from chatree_deka.state import TrialState
from chatree_deka.agents import evaluator_agent, judge_agent
from training.reward import compute_reward


def _summarize_compliance(role: str, turns: list[dict]) -> dict:
    """
    Summarize compliance for a role across their turns.
    
    Calls evaluator_agent.evaluate_turns() for fresh evaluation, then combines with
    any pre-computed valid flags from the transcript to compute a compliance rate.
    """
    evaluations = evaluator_agent.evaluate_turns(role, turns)
    total_turns = len(turns)
    
    # Use evaluator results if available, otherwise fall back to transcript valid flags
    if evaluations:
        valid_turns = sum(1 for valid in evaluations if valid)
    else:
        valid_turns = sum(1 for turn in turns if turn.get("valid") is True)
    
    rate = valid_turns / total_turns if total_turns else 0.0
    return {
        "valid_turns": valid_turns,
        "total_turns": total_turns,
        "rate": rate,
    }


def summary_node(state: TrialState) -> dict:
    transcript = state.get("transcript", [])
    case_id = state.get("case_id", "unknown")
    case_facts = state.get("case_facts", "")
    verdict_text = ""
    for turn in reversed(transcript):
        if turn.get("role") == "judge":
            verdict_text = turn.get("content", "")
            break

    winner = judge_agent.determine_verdict_winner(case_facts, transcript)
    confidence = judge_agent.compute_verdict_confidence(case_facts, transcript, winner)

    prosecutor_turns = [turn for turn in transcript if turn.get("role") == "prosecutor"]
    defender_turns = [turn for turn in transcript if turn.get("role") == "defender"]

    compliance = {
        "prosecutor": _summarize_compliance("prosecutor", prosecutor_turns),
        "defender": _summarize_compliance("defender", defender_turns),
    }

    summary = {
        "case_id": case_id,
        "mode": state.get("mode", "run"),
        "verdict": {
            "winner": winner,
            "confidence": confidence,
            "reasoning": verdict_text,
        },
        "compliance": compliance,
        "episode_reward": None,
    }

    episode_reward: Optional[float] = None
    if state.get("mode") in ("train", "coached"):
        reward_prosecutor = compute_reward(summary, "prosecutor")
        reward_defender = compute_reward(summary, "defender")
        episode_reward = (reward_prosecutor + reward_defender) / 2.0
        summary["episode_reward"] = episode_reward

    return {
        "summary": summary,
        "episode_reward": episode_reward,
    }

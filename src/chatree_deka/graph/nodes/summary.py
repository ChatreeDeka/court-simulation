from __future__ import annotations
from typing import Optional
from chatree_deka.state import TrialState
from chatree_deka.agents import evaluator_agent
from training.reward import compute_reward


def _guess_verdict_winner(ruling: str) -> str:
    text = ruling.lower()
    compromise_markers = [
        "ประนีประนอม",
        "ตกลงกัน",
        "ตกลงร่วม",
        "ทั้งสองฝ่าย",
        "ทั้งโจทก์และจำเลย",
        "ยินยอม",
    ]
    if any(marker in text for marker in compromise_markers):
        return "compromise"

    if "โจทก์" in text and "จำเลย" not in text:
        return "prosecution"
    if "จำเลย" in text and "โจทก์" not in text:
        return "defense"

    prosecution_win = any(
        token in text
        for token in [
            "โจทก์ชนะ",
            "โจทก์ได้รับ",
            "โจทก์ชนะคดี",
            "โจทก์ได้รับคำพิพากษา",
        ]
    )
    defense_win = any(
        token in text
        for token in [
            "จำเลยชนะ",
            "จำเลยได้รับ",
            "จำเลยชนะคดี",
            "จำเลยได้รับคำพิพากษา",
        ]
    )

    if prosecution_win and not defense_win:
        return "prosecution"
    if defense_win and not prosecution_win:
        return "defense"
    if "ไม่" in text and "จำเลย" in text and "โจทก์" not in text:
        return "defense"
    if "ไม่" in text and "โจทก์" in text and "จำเลย" not in text:
        return "prosecution"

    return "compromise"


def _compute_confidence(summary_winner: str, transcript: list[dict]) -> float:
    """
    Compute verdict confidence from transcript validity patterns.
    
    For compromise verdicts, always return 0.5. Otherwise, derive from the fraction
    of valid turns made by the winning side (or opponent for defense consensus).
    """
    if summary_winner == "compromise":
        return 0.5

    role = "prosecutor" if summary_winner == "prosecution" else "defender"
    turns = [turn for turn in transcript if turn.get("role") == role]
    total = len(turns)
    if total == 0:
        return 0.0

    # Count turns with explicit valid=True marker from validation node
    valid = sum(1 for turn in turns if turn.get("valid") is True)
    return valid / total


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
    verdict_text = ""
    for turn in reversed(transcript):
        if turn.get("role") == "judge":
            verdict_text = turn.get("content", "")
            break

    winner = _guess_verdict_winner(verdict_text)
    confidence = _compute_confidence(winner, transcript)

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

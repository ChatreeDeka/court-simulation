from __future__ import annotations
from typing import Literal
from chatree_deka.config import load_config


def compute_reward(summary: dict, role: Literal["prosecutor", "defender"]) -> float:
    verdict_reward = 1.0 if summary["verdict"]["winner"] == role else 0.0
    compliance_reward = summary["compliance"][role]["rate"]
    cfg = load_config()
    weights = cfg.get("training", {}).get("reward_weights", {})
    verdict_weight = float(weights.get("verdict", 0.6))
    compliance_weight = float(weights.get("compliance", 0.4))
    return verdict_weight * verdict_reward + compliance_weight * compliance_reward

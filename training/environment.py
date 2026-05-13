from __future__ import annotations
import json
from pathlib import Path
from loguru import logger
from langgraph.checkpoint.memory import MemorySaver
from chatree_deka.graph.builder import build_graph
from chatree_deka.state import TrialState


def _load_case(case_file: str) -> dict:
    try:
        with open(case_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)
            if isinstance(case_data, list) and case_data:
                case_data = case_data[0]
            return case_data
    except Exception as e:
        logger.error(f"Failed to load case file: {e}")
        return {"case_id": "training_case", "case_facts": ""}


class CourtEnvironment:
    def __init__(self, case_file: str):
        self.case_file = case_file
        self.checkpointer = MemorySaver()
        self.graph = build_graph(checkpointer=self.checkpointer)
        self.thread_config = {"configurable": {"thread_id": "training_session"}}
        self.state = self.reset(case_file)

    def reset(self, case_file: str) -> TrialState:
        case_data = _load_case(case_file)
        state: TrialState = {
            "case_id": case_data.get("case_id", "training_case"),
            "phase": "opening_prosecution",
            "mode": "train",
            "judge_mode": "ai",
            "prosecutor_mode": "ai",
            "defender_mode": "ai",
            "case_facts": case_data.get("case_facts", ""),
            "transcript": [],
            "pending_statement": None,
            "validation_result": None,
            "validation_reason": None,
            "retry_count": 0,
            "max_retries": 2,
            "evaluation_result": None,
            "evaluation_reason": None,
            "objection_pending": False,
            "current_speaker": "prosecutor",
            "episode_reward": None,
            "summary": None,
        }

        for _ in self.graph.stream(state, self.thread_config):
            pass

        self.state = self.graph.get_state(self.thread_config).values
        return self.state

    def step(self, action: str) -> tuple[TrialState, float, bool]:
        if action is not None:
            self.graph.update_state(self.thread_config, {"pending_statement": action})

        for _ in self.graph.stream(None, self.thread_config):
            pass

        current_state = self.graph.get_state(self.thread_config).values
        done = not bool(current_state.next)
        reward = float(current_state.get("episode_reward") or 0.0) if done else 0.0
        return current_state, reward, done

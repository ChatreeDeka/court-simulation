from __future__ import annotations
import json
from pathlib import Path
from loguru import logger
from chatree_deka.config import load_config
from training.environment import CourtEnvironment


def run_training(
    case_file: str | None = None,
    episodes: int | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = None,
) -> None:
    cfg = load_config()
    training_cfg = cfg.get("training", {})

    case_file = case_file or training_cfg.get("case_file", "case-file/1.json")
    episodes = int(episodes if episodes is not None else training_cfg.get("max_episodes", 100))
    checkpoint_path = checkpoint_path or training_cfg.get("checkpoint_path", "training/checkpoints")
    checkpoint_every = int(
        checkpoint_every if checkpoint_every is not None else training_cfg.get("checkpoint_every", 10)
    )

    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = CourtEnvironment(case_file)
    logger.info(f"Starting training for {episodes} episodes")

    for episode in range(1, episodes + 1):
        state, reward, done = env.step("")
        if not done:
            logger.warning("Training episode did not complete as expected")
        logger.info(f"Episode {episode} completed with reward={reward:.4f}")

        if episode % checkpoint_every == 0 or episode == episodes:
            checkpoint_file = checkpoint_dir / f"episode_{episode}.json"
            checkpoint_file.write_text(
                json.dumps({"episode": episode, "reward": reward}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"Checkpoint saved to {checkpoint_file}")

        env = CourtEnvironment(case_file)


if __name__ == "__main__":
    run_training()

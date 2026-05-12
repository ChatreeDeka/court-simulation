from __future__ import annotations
import yaml
from pathlib import Path
from loguru import logger

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
    return {}

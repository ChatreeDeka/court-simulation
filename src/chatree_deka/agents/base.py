from __future__ import annotations
import os
import yaml
from pathlib import Path
from loguru import logger
from litellm import completion

# Load Config
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "config.yaml"

def _get_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {"inference": {"max_tokens": 512, "temperature": 0.3}, "models": {}}

_current_loaded_model = None

def _load_model(model_name: str) -> None:
    """
    Simulates sequential loading to respect 12GB VRAM limits for local models.
    Ensures only one memory-heavy model runs at a time.
    """
    global _current_loaded_model
    if _current_loaded_model != model_name:
        if _current_loaded_model:
            logger.info(f"Unloading {_current_loaded_model} from VRAM...")
        logger.info(f"Loading {model_name} into VRAM sequentially...")
        _current_loaded_model = model_name

def generate(role: str, messages: list[dict]) -> str:
    """
    Generates text using LiteLLM. Handles sequential loads for constraints.
    """
    config = _get_config()
    model_name = config["models"].get(role, "gpt-4o-mini")
    
    _load_model(model_name)
    logger.debug(f"Generating inference for role: {role} using {model_name}")
    
    try:
        response = completion(
            model=model_name,
            messages=messages,
            max_tokens=config["inference"].get("max_tokens", 512),
            temperature=config["inference"].get("temperature", 0.3)
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LiteLLM Generation failed for {role}: {e}")
        return f"Error: Generation failed. ({e})"

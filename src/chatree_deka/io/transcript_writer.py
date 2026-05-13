import json
from pathlib import Path
from loguru import logger

def write_transcript_to_json(transcript: list[dict], case_id: str, output_path: str, summary: dict | None = None) -> None:
    """
    Writes the transcript list and optional post-trial summary to a structured JSON log.
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        log_data = {
            "case_id": case_id,
            "transcript": transcript,
            "summary": summary,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
            
        logger.info(f"Transcript successfully saved to {path}")
    except Exception as e:
        logger.error(f"Failed to write transcript to {output_path}: {e}")

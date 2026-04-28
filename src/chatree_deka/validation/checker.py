from __future__ import annotations
import re
from chatree_deka.validation.schemas import StatementSchema, CheckResult
from chatree_deka.rag.retriever import section_lookup
from loguru import logger

def citation_check(statement: str) -> CheckResult:
    """
    Validates that a statement contains at least one procedural citation (มาตรา or ม.).
    Returns early to prevent DB lookups if no citations exist.
    """
    try:
        # Pydantic validation to enforce the presence of a citation
        parsed = StatementSchema(role="unknown", content=statement)
    except Exception as e:
        logger.warning(f"Citation validation failed: {e}")
        return CheckResult(valid=False, reason="Missing citation. " + str(e))
        
    # Extract distinct section numbers
    matches = re.findall(r'(?:มาตรา|ม\.)\s*(\d+)', statement)
    citations = list(set([int(m) for m in matches]))
    
    if not citations:
        return CheckResult(valid=False, reason="No valid section numbers could be extracted.")
        
    return CheckResult(valid=True, citations_found=citations)

def grounding_check(statement: str, citations: list[int]) -> CheckResult:
    """
    Uses LlamaIndex lookup to verify the cited sections actually exist in the DB.
    """
    if not citations:
        return CheckResult(valid=False, reason="No citations provided for grounding.")
        
    retrieved_content = section_lookup(citations)
    
    if not retrieved_content:
        return CheckResult(valid=False, reason="Could not retrieve the cited sections from corpus.")
        
    return CheckResult(valid=True, reason="Citations exist in reference corpus.")

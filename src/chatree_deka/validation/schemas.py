from __future__ import annotations
import re
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional

class CitationSchema(BaseModel):
    section_number: int

class StatementSchema(BaseModel):
    role: str
    content: str
    
    @field_validator('content')
    @classmethod
    def check_citation(cls, v: str, info: ValidationInfo) -> str:
        # Regex checks for "มาตรา <number>" or "ม. <number>"
        if not re.search(r'(?:มาตรา|ม\.)\s*\d+', v):
            raise ValueError("Statement must contain at least one legal citation (e.g. มาตรา 123 or ม. 123).")
        return v

class CheckResult(BaseModel):
    valid: bool
    reason: Optional[str] = None
    citations_found: list[int] = Field(default_factory=list)

from __future__ import annotations

from functools import lru_cache
import re
from pathlib import Path

import pyarrow.parquet as pq


LAW_PARQUET_PATH = Path(__file__).resolve().parent.parent / "law" / "ccl-00000-of-00001.parquet"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+|[ก-๙]+", _normalize_text(text)))


@lru_cache(maxsize=1)
def _load_law_rows() -> list[dict]:
    table = pq.read_table(LAW_PARQUET_PATH)
    return table.to_pylist()


def _iter_law_entries(row: dict) -> list[dict]:
    entries = []
    for field_name in ("relevant_laws", "reference_laws"):
        for item in row.get(field_name, []) or []:
            entries.append(
                {
                    "law_name": item.get("law_name", ""),
                    "section_num": str(item.get("section_num", "")),
                    "section_content": item.get("section_content", ""),
                }
            )
    return entries


def _format_law_entry(entry: dict) -> str:
    law_name = entry.get("law_name", "")
    section_num = entry.get("section_num", "")
    section_content = entry.get("section_content", "")
    return f"{law_name} มาตรา {section_num}: {section_content}".strip()


def _score_row(query_text: str, query_tokens: set[str], row: dict) -> int:
    searchable_parts = [row.get("question", ""), row.get("answer", "")]
    entries = _iter_law_entries(row)
    searchable_parts.extend(_format_law_entry(entry) for entry in entries)
    combined_text = _normalize_text(" ".join(searchable_parts))

    score = sum(1 for token in query_tokens if token in combined_text)

    query_sections = set(re.findall(r"(?:มาตรา|ม\.)\s*(\d+)", _normalize_text(query_text)))
    if query_sections:
        for entry in entries:
            if entry.get("section_num") in query_sections:
                score += 10

    return score

def semantic_search(query: str, top_k: int = 3) -> str:
    """
    Searches the law parquet and returns the most relevant case-law context.
    """
    rows = _load_law_rows()
    query_tokens = _tokenize(query)

    if not rows:
        return "ไม่พบข้อมูลกฎหมายในคลังอ้างอิง"

    ranked_rows = sorted(rows, key=lambda row: _score_row(query, query_tokens, row), reverse=True)
    top_rows = [row for row in ranked_rows[:top_k] if _score_row(query, query_tokens, row) > 0]
    if not top_rows:
        top_rows = ranked_rows[:top_k]

    sections: list[str] = []
    for row in top_rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        law_entries = [_format_law_entry(entry) for entry in _iter_law_entries(row)]

        block_lines = []
        if question:
            block_lines.append(f"คำถามอ้างอิง: {question}")
        if answer:
            block_lines.append(f"คำตอบอ้างอิง: {answer}")
        if law_entries:
            block_lines.append("กฎหมายที่เกี่ยวข้อง:")
            block_lines.extend(f"- {entry}" for entry in law_entries)

        if block_lines:
            sections.append("\n".join(block_lines))

    return "\n\n".join(sections) if sections else "ไม่พบข้อมูลกฎหมายที่ตรงกับคำค้น"

def section_lookup(section_numbers: list[int]) -> list[str]:
    """
    Looks up specific section numbers from the law parquet.
    """
    rows = _load_law_rows()
    wanted_sections = {str(num) for num in section_numbers}
    results = []

    for row in rows:
        for entry in _iter_law_entries(row):
            if entry.get("section_num") in wanted_sections:
                results.append(_format_law_entry(entry))

    return results

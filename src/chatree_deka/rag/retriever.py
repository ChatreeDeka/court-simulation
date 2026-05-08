from __future__ import annotations

from chatree_deka.rag.chroma_store import lookup_sections, search

def semantic_search(query: str, top_k: int = 3) -> str:
    """
    Searches the Chroma vector store and returns the most relevant case-law context.
    """
    matches = search(query, top_k=top_k)
    if not matches:
        return "ไม่พบข้อมูลกฎหมายในคลังอ้างอิง"

    sections: list[str] = []
    for match in matches:
        document = str(match.get("document", "")).replace("passage: ", "", 1).strip()
        metadata = match.get("metadata", {}) or {}
        law_name = metadata.get("law_name", "")
        section_num = metadata.get("section_num", "")
        if document:
            header = f"{law_name} มาตรา {section_num}".strip()
            sections.append(f"{header}: {document}" if header else document)

    return "\n\n".join(sections) if sections else "ไม่พบข้อมูลกฎหมายที่ตรงกับคำค้น"

def section_lookup(section_numbers: list[int]) -> list[str]:
    """
    Looks up specific section numbers from the Chroma vector store.
    """
    return lookup_sections(section_numbers)

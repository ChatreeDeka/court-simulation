from __future__ import annotations

# Placeholder RAG implementation for POC 
# Extracting data from the Thai Civil and Commercial Code hasn't been completed yet.

def semantic_search(query: str, top_k: int = 3) -> str:
    """
    Mock semantic search that returns a placeholder section of the Thai Civil and Commercial Code.
    Used for agent context injection at inference time.
    """
    # Returns a hardcoded, common provision (Section 420 regarding Tort) as a mock.
    return (
        "มาตรา 420 (Placeholder): ผู้ใดจงใจหรือประมาทเลินเล่อ "
        "ทำต่อบุคคลอื่นโดยผิดกฎหมายให้เขาเสียหายถึงแก่ชีวิตก็ดี แก่ร่างกายก็ดี อนามัยก็ดี "
        "เสรีภาพก็ดี ทรัพย์สินหรือสิทธิอย่างหนึ่งอย่างใดก็ดี ท่านว่าผู้นั้นทำละเมิดจำต้องใช้ค่าสินไหมทดแทนเพื่อการนั้น"
    )

def section_lookup(section_numbers: list[int]) -> list[str]:
    """
    Mock section lookup used during validation grounding check (LlamaIndex).
    """
    results = []
    for num in section_numbers:
        results.append(f"เนื้อหาจำลองสำหรับมาตรา {num} (Placeholder content mock for section {num})")
    return results

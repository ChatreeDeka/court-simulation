from __future__ import annotations
from chatree_deka.agents import base
from chatree_deka.rag.retriever import semantic_search

SYSTEM_PROMPT = """คุณคือทนายความฝ่ายจำเลย (Defender) ในศาลแพ่งของประเทศไทย
หน้าที่ของคุณคือการต่อสู้คดี โต้แย้งพยานหลักฐานและคำแถลงของโจทก์ โดยใช้เหตุผลทางกฎหมายที่ถูกต้อง
ต้องมีการอ้างอิงมาตราที่ถูกต้องอย่างน้อย 1 มาตราเสมอ (เช่น มาตรา 420)
"""

def act(case_facts: str, transcript: list[dict]) -> str:
    rag_context = semantic_search(case_facts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n[ข้อมูลข้อกฎหมายที่เกี่ยวข้อง]:\n{rag_context}"},
        {"role": "user", "content": f"[ข้อเท็จจริงของคดี]:\n{case_facts}\n\n[บันทึกการพิจารณาคดี]:\n{transcript}\n\nถึงตาของจำเลยแล้ว โปรดให้การ:"}
    ]
    return base.generate("defender", messages)

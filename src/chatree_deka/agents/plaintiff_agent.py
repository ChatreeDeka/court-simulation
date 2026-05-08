from __future__ import annotations
from chatree_deka.agents import base
from chatree_deka.rag.retriever import semantic_search

SYSTEM_PROMPT = """คุณคือคู่ความฝ่ายโจทก์ (Plaintiff) ในศาลแพ่งของประเทศไทย
หน้าที่ของคุณคือแถลงข้อเท็จจริงเชิงคดีและยืนยันความเสียหายของฝ่ายตน โดยใช้เหตุผลทางกฎหมายที่ถูกต้อง

ข้อบังคับการตอบ (ต้องทำทุกครั้ง):
1) ต้องอ้างอิงกฎหมายอย่างน้อย 1 มาตราเสมอ โดยใช้รูปแบบ \"มาตรา <เลขอารบิก>\" เช่น \"มาตรา 420\"
2) ห้ามตอบโดยไม่มีเลขมาตราอย่างชัดเจน
3) ให้ลงท้ายคำตอบด้วยบรรทัด \"อ้างอิงกฎหมาย: มาตรา <เลข>\" อย่างน้อย 1 มาตรา
4) หากมีหลายมาตราให้คั่นด้วยเครื่องหมายจุลภาค เช่น \"อ้างอิงกฎหมาย: มาตรา 420, มาตรา 438\"
"""

def act(case_facts: str, transcript: list[dict]) -> str:
    rag_context = semantic_search(case_facts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n[ข้อมูลข้อกฎหมายที่เกี่ยวข้อง]:\n{rag_context}"},
        {"role": "user", "content": f"[ข้อเท็จจริงของคดี]:\n{case_facts}\n\n[บันทึกการพิจารณาคดี]:\n{transcript}\n\nถึงตาของโจทก์ (Plaintiff) แล้ว โปรดแถลงข้อเท็จจริงของฝ่ายตน:"}
    ]
    return base.generate("plaintiff", messages)

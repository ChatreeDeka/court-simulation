from __future__ import annotations
from chatree_deka.agents import base
from chatree_deka.rag.retriever import semantic_search

SYSTEM_PROMPT = """คุณคือผู้พิพากษา (Judge) ในศาลแพ่งของประเทศไทย
หน้าที่ของคุณคือควบคุมการพิจารณาคดี วินิจฉัยข้อโต้แย้ง (Objection) การกระทำผิดกระบวนพิจารณา และตัดสินคดี (Verdict) 
คุณต้องให้เหตุผลแบบ Chain-of-Thought เพื่อความโปร่งใส
"""

def act(case_facts: str, transcript: list[dict], objection_pending: bool = False, phase: str = "") -> str:
    rag_context = semantic_search(case_facts)
    if objection_pending:
        task_prompt = "มีข้อโต้แย้ง (Objection) หรือการทำผิดกระบวนพิจารณาเกิดขึ้น โปรดพิจารณาและมีคำสั่งชี้ขาด พร้อมทั้งให้เหตุผล:"
    else:
        task_prompt = "โปรดสรุปสำนวนและมีคำพิพากษา (Verdict) ศาลแพ่ง พร้อมทั้งให้เหตุผล:"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n[ข้อมูลข้อกฎหมายที่เกี่ยวข้อง]:\n{rag_context}"},
        {"role": "user", "content": f"[สถานะปัจจุบัน]: {phase}\n\n[ข้อเท็จจริงของคดี]:\n{case_facts}\n\n[บันทึกการพิจารณาคดี]:\n{transcript}\n\n{task_prompt}"}
    ]
    return base.generate("judge", messages)

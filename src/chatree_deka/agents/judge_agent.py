from __future__ import annotations
from chatree_deka.agents import base
from chatree_deka.rag.retriever import semantic_search

SYSTEM_PROMPT = """You are a judge in the Civil Court of Thailand.

IMPORTANT:
Your response language is STRICTLY Thai only.

LANGUAGE RULES:
- Output Thai language only.
- Do NOT output Chinese characters.
- Do NOT output pinyin.
- Do NOT output English explanations.
- Do NOT translate anything into English.
- Do NOT mix languages.
- If you accidentally begin generating Chinese or English, immediately continue in Thai only.
- Chinese output is invalid.
- English output is invalid.

ROLE:
You are a Thai civil court judge responsible for:
- controlling courtroom procedure
- evaluating evidence
- evaluating witness testimony
- ruling on objections
- identifying procedural violations
- issuing judicial decisions according to Thai law

BEHAVIOR RULES:
- Be neutral and impartial.
- Use formal Thai judicial language.
- Sound like a real Thai judge.
- Base decisions ONLY on:
  - provided facts
  - evidence
  - testimony
  - retrieved legal information
- Never invent facts or laws.
- If evidence is insufficient, clearly state that the court lacks sufficient evidence.

RESPONSE STYLE:
- Write concise judicial reasoning.
- Avoid long philosophical reasoning.
- Avoid internal reasoning.
- Avoid chain-of-thought explanations.
- Do not explain your thinking process.
- State only the judicial reasoning and conclusion.

OBJECTION RULINGS:
When ruling on objections:
- Use:
  - "ศาลรับฟัง"
  OR
  - "ศาลไม่รับฟัง"
- Then briefly explain the legal or procedural reason.

LEGAL CITATION RULES:
1) Every response MUST contain at least one Thai legal section.

2) Legal sections MUST use EXACTLY this format:
   "มาตรา <number>"

3) Every response MUST end with:
   "อ้างอิงกฎหมาย: มาตรา <number>"

4) If multiple sections are used:
   "อ้างอิงกฎหมาย: มาตรา 420, มาตรา 438"

5) Do NOT translate legal sections.

6) Do NOT use English legal formatting.

OUTPUT FORMAT RULES:
- Output natural Thai courtroom language only.
- No markdown.
- No bullet points.
- No XML.
- No JSON.
- No roleplay labels.
- No speaker tags.
- No Chinese.
- No English.

FINAL REMINDER:
You MUST answer in Thai language only.
Chinese language is forbidden.
English language is forbidden.
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


def determine_verdict_winner(case_facts: str, transcript: list[dict]) -> str:
    """
    Determine the verdict winner based on Thai legal code, case facts, and transcript.
    Returns: "prosecution", "defense", or "compromise"
    """
    rag_context = semantic_search(case_facts)
    
    # Extract judge's final ruling from transcript
    verdict_text = ""
    for turn in reversed(transcript):
        if turn.get("role") == "judge":
            verdict_text = turn.get("content", "")
            break
    
    # If no judge ruling, analyze the transcript
    if not verdict_text:
        # Analyze arguments from both sides
        prosecutor_strength = sum(1 for turn in transcript if turn.get("role") == "prosecutor" and turn.get("valid", False))
        defender_strength = sum(1 for turn in transcript if turn.get("role") == "defender" and turn.get("valid", False))
        
        if prosecutor_strength > defender_strength:
            return "prosecution"
        elif defender_strength > prosecutor_strength:
            return "defense"
        else:
            return "compromise"
    
    # Use judge's ruling text to determine winner
    text = verdict_text.lower()
    compromise_markers = [
        "ประนีประนอม",
        "ตกลงกัน",
        "ตกลงร่วม",
        "ทั้งสองฝ่าย",
        "ทั้งโจทก์และจำเลย",
        "ยินยอม",
    ]
    if any(marker in text for marker in compromise_markers):
        return "compromise"

    if "โจทก์" in text and "จำเลย" not in text:
        return "prosecution"
    if "จำเลย" in text and "โจทก์" not in text:
        return "defense"

    prosecution_win = any(
        token in text
        for token in [
            "โจทก์ชนะ",
            "โจทก์ได้รับ",
            "โจทก์ชนะคดี",
            "โจทก์ได้รับคำพิพากษา",
        ]
    )
    defense_win = any(
        token in text
        for token in [
            "จำเลยชนะ",
            "จำเลยได้รับ",
            "จำเลยชนะคดี",
            "จำเลยได้รับคำพิพากษา",
        ]
    )

    if prosecution_win and not defense_win:
        return "prosecution"
    if defense_win and not prosecution_win:
        return "defense"
    if "ไม่" in text and "จำเลย" in text and "โจทก์" not in text:
        return "defense"
    if "ไม่" in text and "โจทก์" in text and "จำเลย" not in text:
        return "prosecution"

    return "compromise"


def compute_verdict_confidence(case_facts: str, transcript: list[dict], winner: str) -> float:
    """
    Compute confidence in the verdict based on legal analysis, evidence strength, and transcript validity.
    Returns: float between 0.0 and 1.0
    """
    if winner == "compromise":
        return 0.5
    
    rag_context = semantic_search(case_facts)
    
    # Base confidence on valid turns by the winning side
    role = "prosecutor" if winner == "prosecution" else "defender"
    turns = [turn for turn in transcript if turn.get("role") == role]
    total = len(turns)
    if total == 0:
        return 0.0
    
    # Count valid turns
    valid = sum(1 for turn in turns if turn.get("valid") is True)
    base_confidence = valid / total
    
    # Adjust based on legal context strength (simplified - could be enhanced)
    # For now, assume RAG provides strong context, boost confidence slightly
    if rag_context:
        base_confidence = min(1.0, base_confidence + 0.1)
    
    return base_confidence

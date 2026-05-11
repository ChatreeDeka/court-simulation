import re
import json
import fitz
import requests
import hashlib

from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any
from pythainlp.util import normalize
from pythainlp.tokenize import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# CONFIG
# =========================
MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/chat"

ENG_COLUMNS = [
    "judgment_number",
    "plaintiff",
    "defendant",
    "court_type",
    "judgment_date",
    "legal_issues",
    "plaintiff_claim",
    "defendant_testimony",
    "plaintiff_evidence",
    "defendant_evidence",
    "court_accepted_evidence",
    "related_laws",
    "court_reasoning",
    "judgment",
    "damages_awarded",
    "undisputed_facts"
]

SUMMARY_FIELDS = {
    "legal_issues",
    "plaintiff_claim",
    "defendant_testimony",
    "plaintiff_evidence",
    "defendant_evidence",
    "court_accepted_evidence",
    "court_reasoning",
    "judgment",
    "undisputed_facts"
}
CASE_START_PATTERN = r"(?:คำพิพากษาศาลฎีกาที่|คําพิพากษาศาลฎีกาที่)"


# =========================
# EVIDENCE SCHEMA
# =========================
VALID_EVIDENCE_TYPES = {
    "contract",
    "receipt",
    "transfer_slip",
    "financial_record",
    "medical_record",
    "photo",
    "video",
    "audio",
    "chat_log",
    "email",
    "document",
    "official_document",
    "police_report",
    "expert_opinion",
    "witness_testimony",
    "forensic_report",
    "inspection_report",
    "tax_document",
    "employment_record",
    "court_record",
    "identity_document",
    "property_document",
    "digital_record",
    "other"
}

# Map common hallucinations to valid types
EVIDENCE_TYPE_MAP = {
    "statement": "witness_testimony",
    "testimony": "witness_testimony",
    "claim": "witness_testimony",
    "verbal_statement": "witness_testimony",
    "bank_transfer": "transfer_slip",
    "payment_receipt": "receipt",
}


# =========================
# STRUCT HELPERS
# =========================
def make_field(
    value: str = "Not specified",
    confidence: float = 0.0,
    source_text: str = ""
) -> Dict[str, Any]:
    if isinstance(value, (dict, list)):
        out_value = value
    else:
        out_value = value.strip() if isinstance(value, str) else str(value)

    source_text = source_text.strip() if isinstance(source_text, str) else str(source_text)
    confidence = float(max(0.0, min(1.0, confidence)))
    if isinstance(out_value, (dict, list)):
        final_value = out_value if out_value else "Not specified"
    else:
        final_value = out_value if out_value else "Not specified"

    return {
        "value": final_value,
        "confidence": round(confidence, 4),
        "source_text": source_text
    }


def safe_empty_row() -> Dict[str, Dict[str, Any]]:
    return {col: make_field() for col in ENG_COLUMNS}


def extract_value(row: Dict[str, Dict[str, Any]], key: str) -> str:
    field = row.get(key, {})
    val = field.get("value", "Not specified")
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        try:
            return "\n".join([str(x).strip() for x in val if x is not None])
        except Exception:
            return str(val)
    return str(val)


def extract_conf(row: Dict[str, Dict[str, Any]], key: str) -> float:
    field = row.get(key, {})
    return float(field.get("confidence", 0.0))


def extract_source(row: Dict[str, Dict[str, Any]], key: str) -> str:
    field = row.get(key, {})
    src = field.get("source_text", "")
    return src.strip() if isinstance(src, str) else str(src)


# =========================
# TEXT / PDF
# =========================
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts)


def clean_thai_text(text: str) -> str:
    text = normalize(text)
    text = text.replace("\x0c", " ")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_cases(full_text: str) -> List[str]:
    text = clean_thai_text(full_text)
    matches = list(re.finditer(CASE_START_PATTERN, text))

    if not matches:
        return [text] if text.strip() else []

    cases = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        case_text = text[start:end].strip()
        if case_text:
            cases.append(case_text)
    return cases


def prepare_case_text(case_text: str) -> str:
    case_text = clean_thai_text(case_text)
    try:
        sents = sent_tokenize(case_text, engine="newmm")
        sents = [s.strip() for s in sents if s.strip()]
        return "\n".join(sents)
    except Exception:
        return case_text


def split_case_into_chunks(case_text: str, chunk_size: int = 3200, chunk_overlap: int = 300) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(case_text)


# =========================
# EXACT EXTRACTION HELPERS
# =========================

def extract_judgment_number_quick(text: str) -> str:
    m = re.search(r"(?:คำพิพากษาศาลฎีกาที่|คําพิพากษาศาลฎีกาที่)\s*([^\s]+)", text)
    return m.group(1).strip() if m else ""


def case_fingerprint_from_text(case_text: str) -> str:
    jn = extract_judgment_number_quick(case_text)
    if jn:
        return f"judgment_number:{jn}"

    normalized = clean_thai_text(case_text)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"raw_sha1:{digest}"


def extract_judgment_number_info(text: str) -> Dict[str, Any]:
    m = re.search(r"((?:คำพิพากษาศาลฎีกาที่|คําพิพากษาศาลฎีกาที่)\s*([^\s]+))", text)
    if not m:
        return make_field()
    return make_field(
        value=m.group(2).strip(),
        confidence=0.99,
        source_text=m.group(1).strip()
    )


def extract_court_type_info(text: str) -> Dict[str, Any]:
    for court in ["ศาลฎีกา", "ศาลอุทธรณ์", "ศาลชั้นต้น"]:
        if court in text:
            return make_field(value=court, confidence=0.98, source_text=court)
    return make_field()


def extract_related_laws_info(text: str) -> Dict[str, Any]:
    laws = re.findall(r"มาตรา\s*\d+(?:/\d+)?", text)
    laws = list(dict.fromkeys([law.strip() for law in laws]))
    if not laws:
        return make_field()
    return make_field(
        value=" ; ".join(laws),
        confidence=0.97,
        source_text=" ; ".join(laws)
    )


def extract_explicit_thai_date_info(text: str) -> Dict[str, Any]:
    thai_months = {
        "มกราคม": "01",
        "กุมภาพันธ์": "02",
        "มีนาคม": "03",
        "เมษายน": "04",
        "พฤษภาคม": "05",
        "มิถุนายน": "06",
        "กรกฎาคม": "07",
        "สิงหาคม": "08",
        "กันยายน": "09",
        "ตุลาคม": "10",
        "พฤศจิกายน": "11",
        "ธันวาคม": "12"
    }

    m = re.search(
        r"((\d{1,2})\s+(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)\s+(\d{4}))",
        text
    )
    if not m:
        return make_field()

    original = m.group(1)
    day = m.group(2)
    month_th = m.group(3)
    year_be = m.group(4)
    year_ce = int(year_be) - 543
    month = thai_months[month_th]

    return make_field(
        value=f"{year_ce:04d}-{month}-{int(day):02d}",
        confidence=0.99,
        source_text=original
    )


# =========================
# CLEANUP HELPERS
# =========================
def remove_chinese_chars(text: str) -> str:
    # operate only on strings; pass through lists/dicts unchanged
    if not text or text == "Not specified":
        return text
    if not isinstance(text, str):
        return text
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "Not specified"


def deduplicate_lines(text: str) -> str:
    # If given a non-string (list/dict), attempt to deduplicate list items
    if not text or text == "Not specified":
        return text
    if isinstance(text, list):
        seen = []
        for item in text:
            s = str(item).strip()
            if s and s != "ไม่มีข้อมูล" and s not in seen:
                seen.append(s)
        return seen if seen else "Not specified"
    if not isinstance(text, str):
        return str(text)

    lines = [x.strip() for x in re.split(r"[\n;|]+", text) if x.strip()]
    seen = []
    for line in lines:
        if line not in seen and line != "ไม่มีข้อมูล":
            seen.append(line)

    return "\n".join(seen) if seen else "Not specified"


def normalize_evidence(text: str) -> str:
    # Accept lists directly (structured evidence)
    if not text or text == "Not specified":
        return text
    if isinstance(text, list):
        return text
    if not isinstance(text, str):
        return str(text)

    items = re.split(r"[\n;,]+", text)

    cleaned = []
    seen = set()

    for item in items:
        item = item.strip()

        if not item:
            continue

        if len(item) < 3:
            continue

        if item in seen:
            continue

        seen.add(item)
        cleaned.append(item)

    return "\n".join(cleaned) if cleaned else "Not specified"


def shorten_legal_issues(text: str) -> str:
    if not text or text == "Not specified":
        return text

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    lines = lines[:3]
    result = "\n".join(lines)

    if len(result) > 300:
        result = result[:300].strip()

    return result if result else "Not specified"


def clean_party_name(text: str) -> str:
    if not text or text == "Not specified":
        return "Not specified"
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_source_text(text: str, max_len: int = 500) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len].strip()


def normalize_evidence_item(item: Any) -> Dict[str, Any]:
    """
    Normalize an evidence item to ensure it has valid type and description.
    Maps hallucinated types to valid ones; returns None if invalid.
    """
    if not isinstance(item, dict):
        return None

    etype = item.get("type", "").strip().lower()
    desc = item.get("description", "").strip()

    # Map hallucinated types to valid ones
    if etype in EVIDENCE_TYPE_MAP:
        etype = EVIDENCE_TYPE_MAP[etype]

    # Validate type
    if etype not in VALID_EVIDENCE_TYPES:
        etype = "other"

    # Require description
    if not desc:
        return None

    return {
        "type": etype,
        "description": desc
    }


# =========================
# OLLAMA
# =========================
def call_ollama(messages: List[Dict], model_name: str = MODEL_NAME, num_ctx: int = 4096, num_predict: int = 1600) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_ctx": num_ctx,
            "num_predict": num_predict
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


# =========================
# PROMPTS
# =========================
def build_extraction_system_prompt() -> str:
    fields_text = ", ".join(ENG_COLUMNS)
    return f"""
You are a Thai Supreme Court judgment extraction assistant.

You will receive text from ONE Thai court case chunk only.

Return ONLY valid JSON with this structure:
{{
  "judgment_number": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "plaintiff": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "defendant": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "court_type": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "judgment_date": {{"value": "...", "confidence": 0.0, "source_text": "..."}},

  "legal_issues": {{"value": "...", "confidence": 0.0, "source_text": "..."}},

  "plaintiff_claim": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "defendant_testimony": {{"value": "...", "confidence": 0.0, "source_text": "..."}},

    "plaintiff_evidence": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
    "defendant_evidence": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
    "court_accepted_evidence": {{"value": "...", "confidence": 0.0, "source_text": "..."}},

  "related_laws": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "court_reasoning": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "judgment": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "damages_awarded": {{"value": "...", "confidence": 0.0, "source_text": "..."}},
  "undisputed_facts": {{"value": "...", "confidence": 0.0, "source_text": "..."}}
}}

Preferred evidence structure (model should output evidence as arrays of objects):

"evidence": {{
    "plaintiff_evidence": [
        {{"type": "contract", "description": "สัญญากู้ยืมเงินลงวันที่ 5 มกราคม 2565"}},
        {{"type": "bank_transfer", "description": "สลิปโอนเงินจำนวน 500,000 บาท"}}
    ],
    "defendant_evidence": [
        {{"type": "payment_receipt", "description": "ใบเสร็จชำระหนี้บางส่วน"}}
    ],
    "witness_testimony": [
        {{"witness": "นาย ก.", "description": "ยืนยันว่าจำเลยลงลายมือชื่อในสัญญา"}}
    ]
}}

Note: Evidence objects MUST include `type` (or `witness` for witness entries) and `description`. Do NOT include `source_text` or `confidence` in the returned JSON — keep those internally for your reasoning but omit them from the output.

Rules for evidence fields:

1. Evidence MUST be returned as JSON array:
   [
     {{
       "type": "...",
       "description": "..."
     }}
   ]

2. type MUST be ONE of these exact strings (case-insensitive):
   contract, receipt, transfer_slip, financial_record, medical_record,
   photo, video, audio, chat_log, email, document, official_document,
   police_report, expert_opinion, witness_testimony, forensic_report,
   inspection_report, tax_document, employment_record, court_record,
   identity_document, property_document, digital_record, other

3. NEVER invent new evidence types.

4. If evidence is only a verbal assertion or statement without physical proof,
   use type: "witness_testimony"

5. If no evidence is explicitly mentioned, return an empty array: []

6. For narrative Thai text that describes facts without specific physical evidence,
   consider it witness_testimony or move it to plaintiff_claim/defendant_testimony.

Rules:
1. confidence must be a number from 0.0 to 1.0.
2. source_text must be the shortest direct supporting text span from the provided chunk.
3. If a field is not explicitly stated or cannot be reliably inferred:
   - value = "Not specified"
   - confidence = 0.0
   - source_text = ""
4. Use Thai only for content fields, except judgment_date which must be YYYY-MM-DD only if explicitly stated.
5. Do not invent facts.
6. Do not use markdown.
7. Do not output anything except JSON.
8. Do not merge multiple cases.
9. Keep factual fields short and precise.
10. For narrative fields, summarize briefly and faithfully.
11. Extract evidence separately from claims whenever possible.
12. plaintiff_evidence: evidence supporting the plaintiff, such as contracts, receipts, medical certificates, photos, chat logs, witness testimony, transfer slips, and police reports.
13. defendant_evidence: evidence supporting the defendant.
14. court_accepted_evidence: evidence explicitly relied upon or accepted by the court in its reasoning.
15. Keep evidence concise and list-like.

Field list:
{fields_text}
""".strip()


def build_extraction_user_prompt(chunk_text: str) -> str:
    return f"""
Extract the required fields from the following Thai court case text chunk.

Case text:
{chunk_text[:5500]}
""".strip()


def build_summary_system_prompt(field_name: str) -> str:
    return f"""
You are summarizing one extracted legal field from a Thai Supreme Court case.

Field name: {field_name}

Return ONLY valid JSON:
{{
  "value": "...",
  "confidence": 0.0,
  "source_text": "..."
}}

Rules:
1. Summarize only the provided content for this field.
2. Keep important legal meaning.
3. Keep it concise.
4. Do not invent facts.
5. confidence must be 0.0 to 1.0.
6. source_text must be the most relevant supporting text from the provided content.
7. If content is missing, return:
   {{"value":"Not specified","confidence":0.0,"source_text":""}}
8. No markdown. JSON only.
""".strip()


def build_summary_user_prompt(field_name: str, text: str, source_text: str) -> str:
    return f"""
Please summarize this extracted field faithfully and concisely.

Field: {field_name}

Current content:
{text[:7000]}

Supporting text:
{source_text[:3000]}
""".strip()


# =========================
# PARSE JSON OUTPUT
# =========================
def normalize_field_obj(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        value = obj.get("value", "Not specified")
        confidence = obj.get("confidence", 0.0)
        source_text = str(obj.get("source_text", "")).strip()

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        # If the value is structured (list/dict), pass it through
        return make_field(value=value, confidence=confidence, source_text=source_text)

    if isinstance(obj, list):
        # list of evidence objects or strings -> keep as structured value
        return make_field(value=obj, confidence=0.0, source_text="")

    if isinstance(obj, str):
        return make_field(value=obj, confidence=0.5 if obj.strip() and obj.strip() != "Not specified" else 0.0, source_text="")

    return make_field()


def extract_json_block(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_json_output(text: str) -> Dict[str, Dict[str, Any]]:
    raw = extract_json_block(text)

    try:
        data = json.loads(raw)
    except Exception:
        return safe_empty_row()

    # If the model returns a nested `evidence` object, merge those subkeys
    if isinstance(data, dict) and "evidence" in data and isinstance(data["evidence"], dict):
        for k, v in data["evidence"].items():
            # place evidence arrays into top-level keys so downstream code sees them
            data[k] = v

    row = safe_empty_row()
    for col in ENG_COLUMNS:
        raw_val = data.get(col, make_field())
        # If raw_val is a list of dicts, remove internal keys that should not be in final output
        if isinstance(raw_val, list):
            cleaned_list = []
            for item in raw_val:
                if isinstance(item, dict):
                    item_copy = {kk: vv for kk, vv in item.items() if kk not in ("source_text", "confidence")}
                    cleaned_list.append(item_copy)
                else:
                    cleaned_list.append(item)
            row[col] = normalize_field_obj(cleaned_list)
        else:
            row[col] = normalize_field_obj(raw_val)
    return row


def fallback_extract(chunk_text: str) -> Dict[str, Dict[str, Any]]:
    row = safe_empty_row()
    row["judgment_number"] = extract_judgment_number_info(chunk_text)
    row["court_type"] = extract_court_type_info(chunk_text)
    row["related_laws"] = extract_related_laws_info(chunk_text)
    row["judgment_date"] = extract_explicit_thai_date_info(chunk_text)

    if "จำเลยไม่ยื่นคำให้การ" in chunk_text:
        row["defendant_testimony"] = make_field(
            value="จำเลยไม่ยื่นคำให้การ",
            confidence=0.98,
            source_text="จำเลยไม่ยื่นคำให้การ"
        )

    return row


def extract_from_chunk(chunk_text: str, model_name: str = MODEL_NAME) -> Dict[str, Dict[str, Any]]:
    system_prompt = build_extraction_system_prompt()
    user_prompt = build_extraction_user_prompt(chunk_text)

    try:
        raw = call_ollama(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model_name=model_name,
            num_ctx=4096,
            num_predict=1800
        )
        parsed = parse_json_output(raw)
        return parsed
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return fallback_extract(chunk_text)


# =========================
# MERGE CHUNKS INTO 1 CASE
# =========================
def combine_field(existing: Dict[str, Any], new_field: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    ex_val_raw = existing.get("value", "Not specified")
    ex_conf = float(existing.get("confidence", 0.0))
    ex_src = existing.get("source_text", "").strip()

    new_val_raw = new_field.get("value", "Not specified")
    new_conf = float(new_field.get("confidence", 0.0))
    new_src = new_field.get("source_text", "").strip()

    # Handle structured list values (evidence arrays)
    if isinstance(ex_val_raw, list) or isinstance(new_val_raw, list):
        ex_list = ex_val_raw if isinstance(ex_val_raw, list) else ([] if ex_val_raw == "Not specified" else [ex_val_raw])
        new_list = new_val_raw if isinstance(new_val_raw, list) else ([] if new_val_raw == "Not specified" else [new_val_raw])

        if not new_list:
            return existing
        if not ex_list:
            return make_field(new_list, new_conf, "")

        # Merge lists, preserving dict items and deduplicating by JSON representation
        seen = set()
        merged = []
        for item in ex_list + new_list:
            try:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            except Exception:
                key = str(item)
            if key not in seen:
                seen.add(key)
                merged.append(item)

        return make_field(merged, max(ex_conf, new_conf) * 0.95, "")

    # Fallback to string handling
    ex_val = str(ex_val_raw).strip()
    new_val = str(new_val_raw).strip()

    if new_val == "Not specified":
        return existing
    if ex_val == "Not specified":
        return make_field(new_val, new_conf, new_src)

    single_value_fields = {
        "judgment_number",
        "plaintiff",
        "defendant",
        "court_type",
        "judgment_date",
        "damages_awarded"
    }

    if field_name in single_value_fields:
        return make_field(new_val, new_conf, new_src) if new_conf > ex_conf else existing

    if new_val == ex_val:
        best_conf = max(ex_conf, new_conf)
        best_src = ex_src if len(ex_src) >= len(new_src) else new_src
        return make_field(ex_val, best_conf, best_src)

    if new_val in ex_val:
        return make_field(ex_val, max(ex_conf, new_conf), ex_src or new_src)

    if ex_val in new_val and len(new_val) > len(ex_val):
        return make_field(new_val, max(ex_conf, new_conf), new_src or ex_src)

    merged_value = deduplicate_lines(f"{ex_val}\n{new_val}")
    merged_conf = max(ex_conf, new_conf) * 0.95
    merged_source = clean_source_text(f"{ex_src}\n{new_src}", max_len=700)
    return make_field(merged_value, merged_conf, merged_source)


def merge_chunk_rows(rows: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    final_row = safe_empty_row()

    for row in rows:
        for col in ENG_COLUMNS:
            final_row[col] = combine_field(final_row[col], row.get(col, make_field()), col)

    return final_row


# =========================
# SUMMARIZATION
# =========================
def summarize_field(field_name: str, field_obj: Dict[str, Any], model_name: str = MODEL_NAME) -> Dict[str, Any]:
    raw_value = field_obj.get("value", "")
    src = field_obj.get("source_text", "").strip()
    conf = float(field_obj.get("confidence", 0.0))

    # If the field is structured (list/dict), skip summarization and return as-is
    if isinstance(raw_value, (list, dict)):
        return make_field(value=raw_value, confidence=conf, source_text=src)

    text = str(raw_value).strip()

    if not text or text == "Not specified":
        return make_field()

    if len(text) < 220:
        return make_field(value=text, confidence=conf, source_text=src)

    system_prompt = build_summary_system_prompt(field_name)
    user_prompt = build_summary_user_prompt(field_name, text, src)

    try:
        raw = call_ollama(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model_name=model_name,
            num_ctx=4096,
            num_predict=600
        )
        parsed = normalize_field_obj(json.loads(extract_json_block(raw)))
        if parsed["value"] == "Not specified":
            return make_field(value=text, confidence=conf, source_text=src)

        final_conf = min(1.0, max(conf * 0.95, parsed["confidence"]))
        final_src = parsed["source_text"] or src
        return make_field(parsed["value"], final_conf, final_src)

    except Exception as e:
        print(f"Summary error on {field_name}: {e}")
        return make_field(value=text, confidence=conf, source_text=src)


# =========================
# FINAL CLEANING
# =========================
def clean_final_row(row: Dict[str, Dict[str, Any]], full_case_text: str, model_name: str = MODEL_NAME) -> Dict[str, Dict[str, Any]]:
    cleaned = deepcopy(safe_empty_row())

    # exact fields override weaker LLM guesses
    exact_overrides = {
        "judgment_number": extract_judgment_number_info(full_case_text),
        "court_type": extract_court_type_info(full_case_text),
        "judgment_date": extract_explicit_thai_date_info(full_case_text),
        "related_laws": extract_related_laws_info(full_case_text),
    }

    for col in ENG_COLUMNS:
        current = deepcopy(row.get(col, make_field()))
        override = exact_overrides.get(col)

        if override and override["value"] != "Not specified":
            cleaned[col] = override
        else:
            cleaned[col] = current

    # party cleanup
    cleaned["plaintiff"]["value"] = clean_party_name(cleaned["plaintiff"]["value"])
    cleaned["defendant"]["value"] = clean_party_name(cleaned["defendant"]["value"])

    # summarize selected narrative fields
    for col in SUMMARY_FIELDS:
        cleaned[col] = summarize_field(col, cleaned[col], model_name=model_name)

    # cleanup narrative text
    narrative_fields = [
        "legal_issues",
        "plaintiff_claim",
        "defendant_testimony",
        "court_reasoning",
        "judgment",
        "undisputed_facts"
    ]

    evidence_fields = [
        "plaintiff_evidence",
        "defendant_evidence",
        "court_accepted_evidence"
    ]

    for col in narrative_fields:
        val = cleaned[col]["value"]
        cleaned[col]["value"] = deduplicate_lines(remove_chinese_chars(val))
        cleaned[col]["source_text"] = clean_source_text(cleaned[col]["source_text"], max_len=700)

    for col in evidence_fields:
        val = cleaned[col]["value"]

        # If the evidence field is a structured list (from the LLM), strip internal keys
        if isinstance(val, list):
            cleaned_items = []
            for it in val:
                if isinstance(it, dict):
                    # remove source_text/confidence and normalize description strings
                    it.pop("source_text", None)
                    it.pop("confidence", None)
                    if "description" in it and isinstance(it["description"], str):
                        it["description"] = remove_chinese_chars(it["description"]).strip()
                    
                    # Normalize and validate evidence type
                    normalized = normalize_evidence_item(it)
                    if normalized:
                        cleaned_items.append(normalized)
                else:
                    # plain strings -> treat as witness_testimony
                    s = remove_chinese_chars(str(it))
                    if s and s != "Not specified":
                        cleaned_items.append({"type": "witness_testimony", "description": s})

            cleaned[col]["value"] = cleaned_items if cleaned_items else []
            cleaned[col]["source_text"] = ""
        else:
            cleaned[col]["value"] = normalize_evidence(
                remove_chinese_chars(val)
            )

            cleaned[col]["source_text"] = clean_source_text(
                cleaned[col]["source_text"],
                max_len=700
            )

    cleaned["legal_issues"]["value"] = shorten_legal_issues(cleaned["legal_issues"]["value"])

    # force defendant testimony if explicitly absent
    if "จำเลยไม่ยื่นคำให้การ" in full_case_text and cleaned["defendant_testimony"]["value"] == "Not specified":
        cleaned["defendant_testimony"] = make_field(
            value="จำเลยไม่ยื่นคำให้การ",
            confidence=0.98,
            source_text="จำเลยไม่ยื่นคำให้การ"
        )

    # final normalization
    final_row = {}
    for col in ENG_COLUMNS:
        final_row[col] = normalize_field_obj(cleaned.get(col, make_field()))

    return final_row


# =========================
# 1 CASE => 1 JSON RECORD
# =========================
def process_one_case(case_text: str, case_no: int, model_name: str = MODEL_NAME) -> Dict[str, Any]:
    print(f"\nProcessing case {case_no}...")

    prepared = prepare_case_text(case_text)
    chunks = split_case_into_chunks(prepared, chunk_size=3200, chunk_overlap=300)
    print(f"  chunks: {len(chunks)}")

    extracted_rows = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"  extracting chunk {idx}/{len(chunks)}")
        row = extract_from_chunk(chunk, model_name=model_name)
        extracted_rows.append(row)

    merged = merge_chunk_rows(extracted_rows)
    final_row = clean_final_row(merged, case_text, model_name=model_name)

    return {
        "case_no": case_no,
        "raw_case_text": clean_source_text(case_text, max_len=5000),
        "fields": {col: final_row[col] for col in ENG_COLUMNS}
    }


# =========================
# SAVE JSON
# =========================
def save_json(records: List[Dict[str, Any]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(records)} case(s) to {output_path}")


def load_existing_records(output_path: str) -> List[Dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Warning: could not read existing output file ({output_path}): {e}")
        return []

def case_text_fingerprint(case_text: str) -> str:
    normalized = clean_thai_text(case_text)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"raw_sha1:{digest}"


def record_fingerprint(record: Dict[str, Any]) -> str:
    fields = record.get("fields", {})

    # support both old shape:
    # {"fields":{"judgment_number":{"value":"..."}}}
    # and new flat shape:
    # {"judgment_number":"..."}
    judgment_number = "Not specified"
    if isinstance(fields, dict) and fields:
        jn_field = fields.get("judgment_number", {})
        if isinstance(jn_field, dict):
            judgment_number = str(jn_field.get("value", "Not specified")).strip()
        else:
            judgment_number = str(jn_field).strip()
    else:
        judgment_number = str(record.get("judgment_number", "Not specified")).strip()

    if judgment_number and judgment_number != "Not specified":
        return f"judgment_number:{judgment_number}"

    raw_case_text = record.get("raw_case_text", "")
    if raw_case_text:
        normalized = clean_thai_text(raw_case_text)
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return f"raw_sha1:{digest}"

    # fallback for records already saved without raw_case_text
    values = []
    for col in ENG_COLUMNS:
        if isinstance(fields, dict) and fields:
            field_val = fields.get(col, "Not specified")
            if isinstance(field_val, dict):
                val = field_val.get("value", "Not specified")
            else:
                val = field_val
        else:
            val = record.get(col, "Not specified")
        values.append(str(val).strip())

    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()
    return f"fields_sha1:{digest}"


def serialize_record_for_output(record: Dict[str, Any]) -> Dict[str, Any]:
    serialized = {"case_no": record.get("case_no")}
    fields = record.get("fields", {})
    for col in ENG_COLUMNS:
        if isinstance(fields, dict) and fields:
            field = fields.get(col, {})
            if isinstance(field, dict):
                serialized[col] = field.get("value", "Not specified")
            else:
                serialized[col] = field if field else "Not specified"
        else:
            serialized[col] = record.get(col, "Not specified")
    return serialized


def append_unique_records(output_json: str, new_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_records = load_existing_records(output_json)
    seen = {record_fingerprint(r) for r in existing_records}

    added_count = 0
    skipped_count = 0
    for record in new_records:
        fp = record_fingerprint(record)
        if fp in seen:
            skipped_count += 1
            continue
        existing_records.append(record)
        seen.add(fp)
        added_count += 1

    for idx, rec in enumerate(existing_records, 1):
        rec["case_no"] = idx

    records_to_save = [serialize_record_for_output(r) for r in existing_records]
    print(f"Added {added_count} new case(s), skipped {skipped_count} duplicate case(s).")
    save_json(records_to_save, output_json)
    return records_to_save


# =========================
# MAIN PIPELINE
# =========================
def pdf_to_legal_json(pdf_path: str, output_json: str, model_name: str = MODEL_NAME):
    print("Extracting text from PDF...")
    full_text = extract_text_from_pdf(pdf_path)

    print("Splitting into cases...")
    cases = split_into_cases(full_text)
    print(f"Found {len(cases)} case(s)")

    existing_records = load_existing_records(output_json)
    seen = {record_fingerprint(r) for r in existing_records}

    all_records = []
    case_counter = len(existing_records) + 1

    for case_text in cases:
        fp = case_fingerprint_from_text(case_text)

        if fp in seen:
            print("Skipping duplicate case (pre-LLM)")
            continue

        try:
            record = process_one_case(case_text, case_counter, model_name=model_name)
            all_records.append(record)

            seen.add(fp)
            case_counter += 1

        except Exception as e:
            print(f"Case {case_counter} failed: {e}")

            failed_fields = safe_empty_row()
            failed_fields["judgment"] = make_field(
                value=f"Processing failed: {e}",
                confidence=0.0,
                source_text=""
            )

            all_records.append({
                "case_no": case_counter,
                "raw_case_text": "",
                "fields": failed_fields
            })

            case_counter += 1

    print(f"Processed {len(all_records)} / {len(cases)} cases")

    append_unique_records(output_json, all_records)
    print("Done.")


def pdf_dir_to_legal_json(data_dir: str, output_json: str, model_name: str = MODEL_NAME):
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_files = sorted([p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdf_files:
        print(f"No PDF files found in: {data_dir}")
        return

    existing_records = load_existing_records(output_json)
    seen = {record_fingerprint(r) for r in existing_records}

    all_records = []
    case_counter = len(existing_records) + 1

    for pdf_file in pdf_files:
        print(f"\n=== Processing file: {pdf_file.name} ===")

        full_text = extract_text_from_pdf(str(pdf_file))
        cases = split_into_cases(full_text)
        print(f"Found {len(cases)} case(s) in {pdf_file.name}")

        for case_text in cases:
            fp = case_fingerprint_from_text(case_text)

            if fp in seen:
                print("Skipping duplicate case (pre-LLM)")
                continue

            try:
                record = process_one_case(case_text, case_counter, model_name=model_name)

                all_records.append(record)
                seen.add(fp)

                case_counter += 1

            except Exception as e:
                print(f"Case {case_counter} failed: {e}")

                failed_fields = safe_empty_row()
                failed_fields["judgment"] = make_field(
                    value=f"Processing failed: {e}",
                    confidence=0.0,
                    source_text=""
                )

                all_records.append({
                    "case_no": case_counter,
                    "raw_case_text": "",
                    "fields": failed_fields
                })

                case_counter += 1

    append_unique_records(output_json, all_records)
    print("Done.")


if __name__ == "__main__":
    pdf_dir_to_legal_json(
        data_dir="data",
        output_json="output.json",
        model_name="qwen2.5:7b-instruct"
    )

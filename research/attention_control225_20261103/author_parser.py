from __future__ import annotations
import re
from typing import Optional

def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall('\\\\boxed\\{([^}]*)\\}', text)
    if boxes:
        content = boxes[-1]
        number = re.search('[-+]?\\d+(?:\\.\\d+)?', content)
        return number.group(0) if number else content.strip()
    numbers = re.findall('[-+]?\\d+(?:\\.\\d+)?', text)
    if numbers:
        return numbers[-1]
    return None

def extract_gold(text: str) -> Optional[str]:
    match = re.search('####\\s*([-+]?\\d+(?:\\.\\d+)?)', text)
    return match.group(1) if match else None

def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()

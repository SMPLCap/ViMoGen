import json
import re
from typing import Any, Tuple


def interpret_vlm_output(raw_text: str) -> Tuple[str, bool]:
    if not raw_text:
        return "", False

    raw_text = raw_text.strip()
    candidates = [raw_text]
    if "```" in raw_text:
        for block in raw_text.split("```"):
            block = block.strip()
            if not block:
                continue
            if block.lower().startswith("json"):
                block = block[4:].strip()
            candidates.append(block)

    def to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes"}:
                return True
            if lowered in {"false", "no"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        analysis = parsed.get("analysis", raw_text)
        matches = to_bool(parsed.get("matches"))
        return analysis, matches

    lowered = raw_text.lower()
    match = re.search(r'"matches"\\s*:\\s*(true|false)', lowered)
    matches = match.group(1) == "true" if match else ("\"matches\"" in lowered and "true" in lowered)
    return raw_text, matches

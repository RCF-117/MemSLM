"""Generic JSON extraction and relaxed parsing helpers for model outputs."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def extract_first_json_block(text: str) -> str:
    """Extract the first JSON object/array block from arbitrary model output."""
    stripped = str(text).strip()
    if not stripped:
        return "{}"
    code_block = re.search(
        r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```",
        stripped,
        flags=re.IGNORECASE,
    )
    if code_block:
        return str(code_block.group(1)).strip()
    if stripped.startswith("{"):
        obj_end = stripped.rfind("}")
        if obj_end > 0:
            return stripped[: obj_end + 1]
    if stripped.startswith("["):
        array_end = stripped.rfind("]")
        if array_end > 0:
            return stripped[: array_end + 1]
    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    obj_start = stripped.find("{")
    obj_end = stripped.rfind("}")
    if obj_start >= 0 and (array_start < 0 or obj_start < array_start) and obj_end > obj_start:
        return stripped[obj_start : obj_end + 1]
    if array_start >= 0 and array_end > array_start:
        return stripped[array_start : array_end + 1]
    if obj_start >= 0 and obj_end > obj_start:
        return stripped[obj_start : obj_end + 1]
    return stripped


def safe_json_loads(text: str) -> Any:
    """Parse JSON text; return None on parse failure."""
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def safe_json_loads_relaxed(text: str) -> Any:
    """Parse JSON robustly, including lightly malformed or Python-literal outputs."""
    raw = str(text).strip()
    if not raw:
        return {}
    parsed = safe_json_loads(raw)
    if parsed is not None:
        return parsed
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    parsed = safe_json_loads(cleaned)
    if parsed is not None:
        return parsed
    collapsed = " ".join(cleaned.split())
    parsed = safe_json_loads(collapsed)
    if parsed is not None:
        return parsed
    for candidate in (raw, cleaned, collapsed):
        try:
            lit = ast.literal_eval(candidate)
            if isinstance(lit, (dict, list)):
                return lit
        except (ValueError, SyntaxError):
            continue
    return {}

from __future__ import annotations

import json
import re
from typing import Any, Optional

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    m = _CODE_FENCE_RE.search(t)
    if m:
        return (m.group(1) or "").strip()
    return t


def safe_json(text: str) -> Optional[Any]:
    """Parse JSON robustly from LLM outputs. Returns Python object or None."""
    if text is None:
        return None
    t = strip_code_fences(text).strip()
    try:
        return json.loads(t)
    except Exception:
        m = _JSON_RE.search(t)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

from __future__ import annotations

import json
import time
from typing import List, Optional
import requests

from .base import ChatMessage, require_env

class GeminiClient:
    """
    Minimal Gemini REST wrapper using Google AI Studio endpoint.

    Endpoint:
      POST {base_url}/models/{model}:generateContent?key=...

    Notes:
      - Gemini expects "contents" with role "user" or "model".
      - We flatten the conversation into a single user prompt for simplicity + stability.
    """
    provider = "gemini"

    def __init__(self, *, api_key_env: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 max_retries: int = 5):
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.max_retries = int(max_retries)

    def chat(self, *, model: str, messages: List[ChatMessage], temperature: float,
             max_tokens: int, timeout_s: int) -> str:
        key = require_env(self.api_key_env)

        m = (model or "").strip()
        if not m.startswith("models/"):
            m = f"models/{m}"
        url = f"{self.base_url}/{m}:generateContent?key={key}"

        prompt_lines = []
        for msg in messages:
            r = (msg.role or "").lower()
            tag = "SYSTEM" if r == "system" else ("USER" if r == "user" else "ASSISTANT")
            prompt_lines.append(f"[{tag}]\n{msg.content}")
        prompt = "\n\n".join(prompt_lines).strip()

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }

        headers = {"Content-Type": "application/json"}
        last_status: Optional[int] = None
        last_body: str = ""
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
                last_status = r.status_code
                last_body = (r.text or "")[:2000]

                if r.status_code in (429, 500, 502, 503, 504):
                    delay_s: float = 1.5 * (attempt + 1)
                    try:
                        data = r.json()
                        details = data.get("error", {}).get("details", [])
                        for d in details:
                            if str(d.get("@type", "")).endswith("RetryInfo") and "retryDelay" in d:
                                rd = d.get("retryDelay")
                                if isinstance(rd, str) and rd.endswith("s"):
                                    delay_s = float(rd[:-1])
                    except Exception:
                        pass
                    time.sleep(delay_s)
                    continue

                r.raise_for_status()
                data = r.json()
                cands = data.get("candidates", [])
                if not cands:
                    return ""
                parts = cands[0].get("content", {}).get("parts", [])
                txt = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
                return (txt or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(
            "Gemini chat failed after retries.\n"
            f"Last HTTP status: {last_status}\n"
            f"Last body (first 2000 chars): {last_body}\n"
            f"Last exception: {last_err}"
        )

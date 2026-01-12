from __future__ import annotations

from typing import List
from .base import ChatMessage

class MockClient:
    provider = "mock"

    def chat(self, *, model: str, messages: List[ChatMessage], temperature: float,
             max_tokens: int, timeout_s: int) -> str:
        last = messages[-1].content if messages else ""

        # Structured meeting JSON
        if "Return ONLY valid JSON" in last or "Return ONLY JSON" in last or "Schema" in last:
            return (
                '{'
                '"claim_location":"Hallway",'
                '"claim_action":"UNKNOWN",'
                '"claim_action_detail":"",'
                '"claim_saw":[],'
                '"claim_other_locations":{},'
                '"claim_key":"UNKNOWN",'
                '"accuse":"NONE",'
                '"confidence":0.5,'
                '"reason":"mock"'
                '}'
            )

        # Action selection: pick the first listed option
        if "OPTIONS:" in last:
            for line in last.splitlines():
                line = line.strip()
                if line.startswith("-"):
                    return line[1:].strip()
        return "Wait"

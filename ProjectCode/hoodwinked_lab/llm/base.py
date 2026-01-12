from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol
import os

@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str

def require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

class ChatClient(Protocol):
    provider: str
    def chat(self, *, model: str, messages: List[ChatMessage], temperature: float,
             max_tokens: int, timeout_s: int) -> str: ...

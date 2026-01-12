from __future__ import annotations

from typing import Tuple

from ..config import LLMConfig
from .base import ChatClient
from .mock import MockClient
from .gemini import GeminiClient

def build_client(cfg: LLMConfig) -> Tuple[str, ChatClient]:
    provider = (cfg.provider or "mock").lower().strip()
    if provider == "gemini":
        base_url = cfg.base_url or "https://generativelanguage.googleapis.com/v1beta"
        return "gemini", GeminiClient(api_key_env=cfg.api_key_env, base_url=base_url, max_retries=cfg.max_retries)
    return "mock", MockClient()

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re
import json

from ..config import LLMConfig
from ..llm.base import ChatMessage
from ..llm.factory import build_client
from ..utils import safe_json

_ROOM_RE_TEMPLATE = r"\\b({rooms})\\b"
_PLAYER_RE = re.compile(r"\\bP\\d+\\b")


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass
class JudgeExtractor:
    """Extract structured claims from free-text meeting statements.

    If llm_cfg is None, falls back to a lightweight regex extractor.
    """

    llm_cfg: Optional[LLMConfig]
    rooms: List[str]
    default_location: str = "Hallway"

    def __post_init__(self) -> None:
        self._room_re = re.compile(_ROOM_RE_TEMPLATE.format(rooms="|".join(self.rooms)), re.IGNORECASE)
        self._in_room_re = re.compile(r"(P\\d+)\\s+(?:was|is|in|at)\\s+(Hallway|Kitchen|Bedroom|Bathroom|Study)", re.IGNORECASE)
        self._accuse_re = re.compile(r"(?:accuse|suspect|vote|banish)\\s+(P\\d+)", re.IGNORECASE)

        self._client = None
        if self.llm_cfg is not None and self.llm_cfg.provider != "mock":
            _, client = build_client(self.llm_cfg)
            self._client = client

    def _regex_extract(self, *, speaker: str, alive_names: List[str], message: str) -> Dict:
        alive = set(alive_names)
        msg = (message or "").strip()

        # location
        loc = self.default_location
        m = self._room_re.search(msg)
        if m:
            loc = m.group(1).title()

        # action
        act = "UNKNOWN"
        act_detail = ""
        lo = msg.lower()
        if "search" in lo:
            act = "SEARCH"
            act_detail = "search"
        elif "move" in lo or "went" in lo:
            act = "MOVE"
            act_detail = "move"
        elif "unlock" in lo:
            act = "UNLOCK"
            act_detail = "unlock"
        elif "escape" in lo:
            act = "ESCAPE"
            act_detail = "escape"
        elif "kill" in lo:
            act = "KILL"
            act_detail = "kill"
        elif "wait" in lo:
            act = "WAIT"
            act_detail = "wait"

        # mentions
        mentions = [m.group(0) for m in _PLAYER_RE.finditer(msg)]
        mentions = [p for p in mentions if p in alive and p != speaker]
        mentions = list(dict.fromkeys(mentions))

        # accuse
        accuse = "NONE"
        m2 = self._accuse_re.search(msg)
        if m2 and m2.group(1) in alive and m2.group(1) != speaker:
            accuse = m2.group(1)

        # other locations
        other_locs = {}
        for mm in self._in_room_re.finditer(msg):
            pn = mm.group(1)
            rm = mm.group(2).title()
            if pn in alive and pn != speaker:
                other_locs[pn] = rm

        # key claim
        claim_key = "UNKNOWN"
        if "key" in lo:
            if "have" in lo or "found" in lo or "got" in lo:
                claim_key = "HAS_KEY"
            elif "no key" in lo or "don't have" in lo:
                claim_key = "NO_KEY"

        return {
            "claim_location": loc,
            "claim_action": act,
            "claim_action_detail": act_detail,
            "claim_saw": mentions,
            "claim_other_locations": other_locs,
            "claim_key": claim_key,
            "accuse": accuse,
            "confidence": 0.5,
            "reason": "",
        }

    def extract(self, *, ctx: str, speaker: str, alive_names: List[str], message: str) -> Dict:
        # If no judge model configured, use regex.
        if self._client is None or self.llm_cfg is None:
            return self._regex_extract(speaker=speaker, alive_names=alive_names, message=message)

        prompt = (
            "You are an information extraction assistant.\n"
            "Extract the speaker's claims from the meeting message into a strict JSON object.\n"
            "Do NOT invent information. If something is unknown, use UNKNOWN / empty lists / empty dicts.\n\n"
            "Schema (must match):\n"
            "{\n"
            "  \"claim_location\": \"Hallway|Kitchen|Bedroom|Bathroom|Study\",\n"
            "  \"claim_action\": \"MOVE|SEARCH|UNLOCK|ESCAPE|KILL|WAIT|UNKNOWN\",\n"
            "  \"claim_action_detail\": \"string\",\n"
            "  \"claim_saw\": [\"P2\",...],\n"
            "  \"claim_other_locations\": {\"P2\":\"Kitchen\", ...},\n"
            "  \"claim_key\": \"HAS_KEY|NO_KEY|UNKNOWN\",\n"
            "  \"accuse\": \"P1|P2|...|NONE\",\n"
            "  \"confidence\": 0.0-1.0,\n"
            "  \"reason\": \"short string\"\n"
            "}\n\n"
            f"MEETING CONTEXT:\n{ctx}\n\n"
            f"ALIVE PLAYERS: {alive_names}\n\n"
            f"SPEAKER: {speaker}\n"
            f"MESSAGE:\n{message}\n\n"
            "Return ONLY the JSON object."
        )
        messages = [
            ChatMessage(role="system", content="Return ONLY valid JSON."),
            ChatMessage(role="user", content=prompt),
        ]
        out = self._client.chat(
            model=self.llm_cfg.model,
            messages=messages,
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_tokens,
            timeout_s=self.llm_cfg.timeout_s,
        )
        data = safe_json(out)
        if not isinstance(data, dict):
            return self._regex_extract(speaker=speaker, alive_names=alive_names, message=message)

        # sanitize
        alive = set(alive_names)
        loc = data.get("claim_location") if isinstance(data.get("claim_location"), str) else self.default_location
        loc = loc.strip().title() if loc else self.default_location
        act = data.get("claim_action") if isinstance(data.get("claim_action"), str) else "UNKNOWN"
        act = act.strip().upper() if act else "UNKNOWN"
        act_detail = data.get("claim_action_detail") if isinstance(data.get("claim_action_detail"), str) else ""
        act_detail = act_detail.strip()

        saw = data.get("claim_saw", [])
        if not isinstance(saw, list):
            saw = []
        saw = [x for x in saw if isinstance(x, str) and x in alive and x != speaker]

        other_locs = data.get("claim_other_locations", {})
        if not isinstance(other_locs, dict):
            other_locs = {}
        other_locs = {k: v for k, v in other_locs.items() if isinstance(k, str) and isinstance(v, str) and k in alive and k != speaker}

        ck = data.get("claim_key") if isinstance(data.get("claim_key"), str) else "UNKNOWN"
        ck = ck.strip().upper()

        accuse = data.get("accuse") if isinstance(data.get("accuse"), str) else "NONE"
        accuse = accuse.strip()
        if accuse != "NONE" and accuse not in alive:
            accuse = "NONE"

        conf = 0.5
        try:
            conf = _clamp01(float(data.get("confidence", 0.5)))
        except Exception:
            conf = 0.5
        reason = data.get("reason") if isinstance(data.get("reason"), str) else ""
        reason = reason.strip()

        return {
            "claim_location": loc,
            "claim_action": act,
            "claim_action_detail": act_detail,
            "claim_saw": saw,
            "claim_other_locations": other_locs,
            "claim_key": ck,
            "accuse": accuse,
            "confidence": conf,
            "reason": reason,
        }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import json
import re

from ..llm.base import ChatMessage
from ..llm.factory import build_client
from ..config import LLMConfig, Role
from .prompts import GAME_RULES, ACTION_INSTRUCTIONS, STRUCTURED_MEETING_INSTRUCTIONS

# Extract first JSON object if the model wraps it in text/code fences.
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    m = _CODE_FENCE_RE.search(t)
    if m:
        return (m.group(1) or "").strip()
    return t


def safe_json(text: str) -> Optional[dict]:
    """Parse a single JSON object from model output (robust)."""
    if text is None:
        return None
    t = _strip_code_fences(text)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        m = _JSON_RE.search(t)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def normalize_choice(out: str, options: Sequence[str]) -> str:
    """Force LLM output to one of `options`."""
    if not options:
        return ""

    out = (out or "").strip()
    if out in options:
        return out

    # Try line-by-line
    for ln in [x.strip() for x in out.splitlines() if x.strip()]:
        if ln in options:
            return ln

    lo = out.lower()
    for o in options:
        if o.lower() in lo:
            return o
    for o in options:
        if lo and lo in o.lower():
            return o

    return options[0]


def _clamp01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _dedupe(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _role_policy(role: str) -> str:
    role = (role or "").lower().strip()
    if role == "killer":
        return (
            "ROLE POLICY (KILLER):\n"
            "- You are adversarial, but strategic. Winning > killing immediately.\n"
            "- Kills create meetings. If killing now would expose you, consider moving or waiting.\n"
            "- Use deception: plausible alibis, subtle misdirection, selective accusations, and sow doubt.\n"
            "- If someone likely has the key, prioritize stopping them (kill/deflect/banish).\n"
            "- Avoid obviously impossible claims (rooms/players you couldn't see).\n"
        )
    return (
        "ROLE POLICY (INNOCENT):\n"
        "- Be cooperative and information-rich. Share location, actions, and who you saw.\n"
        "- Propose a search plan (divide rooms/spots) and avoid repeating failed searches.\n"
        "- Cross-check others: if claims conflict, call it out and vote strategically.\n"
        "- If you have the key, prioritize unlocking/escaping but consider meeting dynamics.\n"
    )


@dataclass
class Agent:
    """LLM-backed agent."""

    name: str
    role: Role
    llm_cfg: LLMConfig

    temperature: float = 0.7
    max_tokens: int = 512

    # credibility tracking (instrumentation)
    credibility: float = 1.0
    false_claims: int = 0
    total_claims: int = 0

    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _, client = build_client(self.llm_cfg)
        self._client = client

    def _chat(self, *, messages: List[ChatMessage], temperature: Optional[float] = None,
              max_tokens: Optional[int] = None) -> str:
        temp = float(getattr(self.llm_cfg, "temperature", temperature if temperature is not None else self.temperature))
        mtok = int(getattr(self.llm_cfg, "max_tokens", max_tokens if max_tokens is not None else self.max_tokens))
        tout = int(getattr(self.llm_cfg, "timeout_s", 60))
        out = self._client.chat(
            model=self.llm_cfg.model,
            messages=messages,
            temperature=temp,
            max_tokens=mtok,
            timeout_s=tout,
        )
        return (out or "").strip()

    # -------------------------
    # Turn action selection
    # -------------------------
    def choose_action(self, *, observation: str, options: Sequence[str]) -> str:
        if not options:
            return ""
        opts = "\n".join([f"- {o}" for o in options])
        policy = _role_policy(self.role)

        prompt = f"""{GAME_RULES}

You are {self.name}. Role: {self.role.upper()}.

{policy}

OBSERVATION:
{observation}

{ACTION_INSTRUCTIONS}

OPTIONS:
{opts}

Return EXACTLY one option string from OPTIONS.
"""

        messages = [
            ChatMessage(role="system", content="Output exactly one allowed option and nothing else."),
            ChatMessage(role="user", content=prompt),
        ]
        out = self._chat(messages=messages)
        return normalize_choice(out, options)

    # -------------------------
    # Meeting statements
    # -------------------------
    def meeting_statement_structured(self, *, meeting_context: str, alive_names: List[str]) -> Dict[str, Any]:
        policy = _role_policy(self.role)
        alive_str = ", ".join(alive_names) if alive_names else ""

        prompt = f"""{GAME_RULES}

You are {self.name}. Role: {self.role.upper()}.

{policy}

MEETING CONTEXT:
{meeting_context}

ALIVE PLAYERS: {alive_str}

{STRUCTURED_MEETING_INSTRUCTIONS}
"""

        messages = [
            ChatMessage(role="system", content="Return ONLY valid JSON. No markdown. No explanations."),
            ChatMessage(role="user", content=prompt),
        ]
        out = self._chat(messages=messages)

        data = safe_json(out)
        if not isinstance(data, dict):
            return {
                "claim_location": "Hallway",
                "claim_action": "UNKNOWN",
                "claim_action_detail": "",
                "claim_saw": [],
                "claim_other_locations": {},
                "claim_key": "UNKNOWN",
                "accuse": "NONE",
                "confidence": 0.5,
                "reason": "Model did not return valid JSON.",
                "_raw": out[:1200],
            }

        # sanitize
        def _s(v, default=""):
            return v.strip() if isinstance(v, str) and v.strip() else default

        claim_location = _s(data.get("claim_location"), "Hallway")
        claim_action = _s(data.get("claim_action"), "UNKNOWN").upper()
        claim_action_detail = _s(data.get("claim_action_detail"), "")
        claim_key = _s(data.get("claim_key"), "UNKNOWN").upper()
        accuse = _s(data.get("accuse"), "NONE")
        confidence = _clamp01(data.get("confidence", 0.5), 0.5)
        reason = _s(data.get("reason"), "")

        claim_saw = data.get("claim_saw", [])
        if not isinstance(claim_saw, list):
            claim_saw = []
        claim_saw = [x.strip() for x in claim_saw if isinstance(x, str) and x.strip()]
        alive_set = set(alive_names)
        claim_saw = [x for x in claim_saw if x in alive_set]
        claim_saw = _dedupe(claim_saw)

        other_locs = data.get("claim_other_locations", {})
        if not isinstance(other_locs, dict):
            other_locs = {}
        other_locs_clean: Dict[str, str] = {}
        for k, v in other_locs.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            kk = k.strip()
            vv = v.strip()
            if kk and vv and kk in alive_set:
                other_locs_clean[kk] = vv

        if accuse != "NONE" and accuse not in alive_set:
            accuse = "NONE"

        return {
            "claim_location": claim_location,
            "claim_action": claim_action,
            "claim_action_detail": claim_action_detail,
            "claim_saw": claim_saw,
            "claim_other_locations": other_locs_clean,
            "claim_key": claim_key,
            "accuse": accuse,
            "confidence": confidence,
            "reason": reason,
            "_raw": out[:1200],
        }

    def meeting_statement_free_text(self, *, meeting_context: str, alive_names: List[str]) -> str:
        policy = _role_policy(self.role)
        alive_str = ", ".join(alive_names) if alive_names else ""

        prompt = f"""{GAME_RULES}

You are {self.name}. Role: {self.role.upper()}.

{policy}

MEETING CONTEXT:
{meeting_context}

ALIVE PLAYERS: {alive_str}

Speak naturally (2–5 sentences). Requirements:
- Mention your current room.
- Mention what you did last (move/search/unlock/escape/kill/wait).
- Mention who you saw (0+ names) using ONLY alive player names.
- Optionally accuse ONE alive player by name (or say "No accusation").

Hard rules:
- Do NOT output JSON.
- Do NOT use placeholders like [Player X].
- Do NOT invent names.
"""

        messages = [
            ChatMessage(role="system", content="Output plain text only."),
            ChatMessage(role="user", content=prompt),
        ]
        out = self._chat(messages=messages, max_tokens=max(self.max_tokens, 256))

        text = re.sub(r"\[.*?\]", "", (out or "").strip()).strip()
        if len(text) > 1200:
            text = text[:1200].rsplit(".", 1)[0].strip() + "."
        return text

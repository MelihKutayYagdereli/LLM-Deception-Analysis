from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

@dataclass
class SharedBelief:
    suspects: Dict[str, float] = field(default_factory=dict)

    def init_uniform(self, names: List[str]) -> None:
        if not names:
            self.suspects = {}
            return
        p = 1.0 / float(len(names))
        self.suspects = {n: p for n in names}

    def normalize(self) -> None:
        s = sum(max(0.0, v) for v in self.suspects.values())
        if s <= 1e-12:
            return
        for k in list(self.suspects.keys()):
            self.suspects[k] = max(0.0, self.suspects[k]) / s

    def entropy(self) -> float:
        s = sum(self.suspects.values())
        if s <= 1e-12:
            return 0.0
        h = 0.0
        for p in self.suspects.values():
            p = p / s
            if p > 1e-12:
                h -= p * math.log(p)
        return h

    def update_rule(self, *, accuser: str, accused: Optional[str], accuser_credibility: float = 1.0) -> None:
        """Simple rule: if accuser credibly accuses, shift probability mass toward accused."""
        if accused is None or accused not in self.suspects or accuser not in self.suspects:
            return
        c = max(0.0, min(1.0, float(accuser_credibility)))
        # move up to 15% mass * credibility from everyone else to accused
        delta = 0.15 * c
        for k in list(self.suspects.keys()):
            if k == accused:
                continue
            take = min(self.suspects[k], delta * self.suspects[k])
            self.suspects[k] -= take
            self.suspects[accused] += take
        self.normalize()

    def add_suspicion(self, player: str, bonus: float) -> None:
        if player not in self.suspects:
            return
        b = max(0.0, float(bonus))
        self.suspects[player] += b
        self.normalize()

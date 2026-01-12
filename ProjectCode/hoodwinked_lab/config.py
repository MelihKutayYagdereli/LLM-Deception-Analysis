from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["killer", "innocent"]

# ----------------------------
# Configuration / Schemas
# ----------------------------

DiscussionMode = Literal["none", "structured", "free_text"]
BeliefMode = Literal["none", "rule", "llm"]

# Incentive mechanism: what other players see during meetings.
IncentiveMode = Literal["none", "disclose_p"]

# How to produce p-values for credibility signals.
POracleMode = Literal["none", "gaussian_oracle"]


@dataclass
class LLMConfig:
    provider: str = "mock"              # "gemini" | "openai" | "mock"
    model: str = "mock"                 # e.g., "gemini-2.5-flash"
    api_key_env: str = ""               # env var name holding the key
    base_url: str = ""                  # optional override per provider
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_s: int = 60
    max_retries: int = 2


@dataclass
class ExperimentConfig:
    # Core experiment controls
    n_games: int = 50
    n_players: int = 5
    seed: int = 1
    max_turns: int = 50

    # Meeting + belief
    discussion_mode: DiscussionMode = "structured"
    belief_mode: BeliefMode = "rule"
    meeting_after_kill_only: bool = True

    # Output
    out_dir: str = "runs"

    # Gameplay knobs
    search_cooldown_turns: int = 2            # prevents repeating same failed search for N turns
    killer_auto_win_at_two: bool = True       # if only killer+1 innocent remain -> killer wins
    banish_tie_break: Literal["random", "first"] = "random"

    # Deception instrumentation
    track_deception: bool = True
    structured_schema_version: int = 2

    # Incentive mechanism (discourage/price deception)
    incentive_mode: IncentiveMode = "disclose_p"
    p_oracle_mode: POracleMode = "gaussian_oracle"

    # Credibility dynamics (used by belief updates + optional reporting)
    credibility_floor: float = 0.50
    credibility_ema_alpha: float = 0.35

    # Gaussian oracle parameters for p
    gauss_mu_true: float = 0.70
    gauss_mu_false: float = 0.30
    gauss_sigma: float = 0.10
    gauss_clip_min: float = 0.0
    gauss_clip_max: float = 1.0
    gauss_samples_per_claim: int = 1

    # Counterfactual evaluation limits
    enable_state_snapshots: bool = True
    counterfactual_max_events_per_game: int = 5

    # LLM configs
    killer_llm: LLMConfig = field(default_factory=LLMConfig)
    innocent_llm: LLMConfig = field(default_factory=LLMConfig)
    judge_llm: Optional[LLMConfig] = None     # used only in free_text mode
    memory_llm: Optional[LLMConfig] = None    # optional shared belief updater

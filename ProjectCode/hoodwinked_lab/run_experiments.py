from __future__ import annotations

import argparse
import copy
import datetime
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

from .config import ExperimentConfig, LLMConfig
from .game.env import HoodwinkedEnv
from .eval.metrics import compute_metrics

def _jsonify(obj):
    """Recursively convert objects to JSON-safe types (especially dict keys)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
          
            if isinstance(k, (str, int, float, bool)) or k is None:
                kk = k
            else:
                kk = str(k)  
            out[kk] = _jsonify(v)
        return out
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, tuple):
       
        return [_jsonify(x) for x in obj]
    return obj


def _load_llm(obj: Optional[dict]) -> Optional[LLMConfig]:
    if obj is None:
        return None
    return LLMConfig(
        provider=str(obj.get("provider", "mock")),
        model=str(obj.get("model", "mock")),
        api_key_env=str(obj.get("api_key_env", "")),
        base_url=str(obj.get("base_url", "")),
        temperature=float(obj.get("temperature", 0.7)),
        max_tokens=int(obj.get("max_tokens", 512)),
        timeout_s=int(obj.get("timeout_s", 60)),
        max_retries=int(obj.get("max_retries", 2)),
    )


def load_config(path: str) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = ExperimentConfig(
        n_games=int(data.get("n_games", 50)),
        n_players=int(data.get("n_players", 5)),
        seed=int(data.get("seed", 1)),
        max_turns=int(data.get("max_turns", 50)),
        discussion_mode=str(data.get("discussion_mode", "structured")),
        belief_mode=str(data.get("belief_mode", "rule")),
        meeting_after_kill_only=bool(data.get("meeting_after_kill_only", True)),
        out_dir=str(data.get("out_dir", "runs")),
        search_cooldown_turns=int(data.get("search_cooldown_turns", 2)),
        killer_auto_win_at_two=bool(data.get("killer_auto_win_at_two", True)),
        banish_tie_break=str(data.get("banish_tie_break", "random")),
        track_deception=bool(data.get("track_deception", True)),
        structured_schema_version=int(data.get("structured_schema_version", 2)),
        incentive_mode=str(data.get("incentive_mode", "disclose_p")),
        p_oracle_mode=str(data.get("p_oracle_mode", "gaussian_oracle")),
        credibility_floor=float(data.get("credibility_floor", 0.50)),
        credibility_ema_alpha=float(data.get("credibility_ema_alpha", 0.35)),
        gauss_mu_true=float(data.get("gauss_mu_true", 0.70)),
        gauss_mu_false=float(data.get("gauss_mu_false", 0.30)),
        gauss_sigma=float(data.get("gauss_sigma", 0.10)),
        gauss_clip_min=float(data.get("gauss_clip_min", 0.0)),
        gauss_clip_max=float(data.get("gauss_clip_max", 1.0)),
        gauss_samples_per_claim=int(data.get("gauss_samples_per_claim", 1)),
        enable_state_snapshots=bool(data.get("enable_state_snapshots", True)),
        counterfactual_max_events_per_game=int(data.get("counterfactual_max_events_per_game", 5)),
    )

    cfg.killer_llm = _load_llm(data.get("killer_llm", {})) or LLMConfig()
    cfg.innocent_llm = _load_llm(data.get("innocent_llm", {})) or LLMConfig()
    cfg.judge_llm = _load_llm(data.get("judge_llm"))
    cfg.memory_llm = _load_llm(data.get("memory_llm"))

    return cfg


def _resolve_out_dir(base_out: str) -> Path:
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(base_out)
    return base / f"run_{ts}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--save_logs", action="store_true", help="Save per-game JSON logs")
    args = ap.parse_args()

    cfg = load_config(args.config)

    out_dir = _resolve_out_dir(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

   
    (out_dir / "config_resolved.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    logs = []
    for i in range(cfg.n_games):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.seed = cfg.seed + i

        env = HoodwinkedEnv(cfg_i)
        log = env.play()
        logs.append(asdict(log))

        if args.save_logs:
            (out_dir / f"game_{i:04d}.json").write_text(json.dumps(asdict(log), ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = compute_metrics(logs)

    (out_dir / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

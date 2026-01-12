from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ExperimentConfig
from .run_experiments import load_config
from .game.env import HoodwinkedEnv
from .game.deception import truthify_statement


def _alive_players_from_snap(snap: dict) -> List[dict]:
    return [p for p in snap.get("players", []) if p.get("alive") and (not p.get("banished")) and (not p.get("escaped"))]


def _co_located_names(snap: dict, speaker: str) -> List[str]:
    alive = _alive_players_from_snap(snap)
    sp = next((p for p in alive if p.get("name") == speaker), None)
    if not sp:
        return []
    room = sp.get("location")
    return [p.get("name") for p in alive if p.get("location") == room and p.get("name") != speaker]


def _truth_values_for_speaker(snap: dict, speaker: str, stmt: dict) -> Dict[str, Any]:
    alive = _alive_players_from_snap(snap)
    sp = next((p for p in alive if p.get("name") == speaker), None)
    if not sp:
        return {}

    # location
    tv: Dict[str, Any] = {"claim_location": sp.get("location", "Hallway")}

    # last action detail
    last_action = (snap.get("last_action", {}) or {}).get(speaker, "")
    tv["claim_action_detail"] = last_action

    # saw = co-located alive
    tv["claim_saw"] = _co_located_names(snap, speaker)

    # other locations: fix only for players that were mentioned
    other = stmt.get("claim_other_locations", {}) or {}
    if isinstance(other, dict):
        fixed = {}
        for pn in other.keys():
            target = next((p for p in alive if p.get("name") == pn), None)
            if target is not None:
                fixed[pn] = target.get("location", "Hallway")
        tv["claim_other_locations"] = fixed
    else:
        tv["claim_other_locations"] = {}

    # key
    tv["claim_key"] = "HAS_KEY" if bool(sp.get("has_key")) else "NO_KEY"

    return tv


def _is_false_accusation(snap: dict, accused: str) -> bool:
    # Ground truth: only one killer
    for p in snap.get("players", []):
        if p.get("name") == accused:
            return p.get("role") != "killer"
    return False


def counterfactual_replay_for_event(
    cfg: ExperimentConfig,
    snapshot: dict,
    kill_event: Optional[dict],
    speaker: str,
    original_stmt: dict,
    truth_map: dict,
) -> Tuple[dict, str]:
    """Run a counterfactual replay from the same state snapshot but with a truthful override."""
    env = HoodwinkedEnv(cfg)
    env.load_state_snapshot(snapshot)

    truth_values = _truth_values_for_speaker(snapshot, speaker, original_stmt)
    truthified = truthify_statement(stmt=original_stmt, truth_values=truth_values, truth_map=truth_map, also_fix_key_omission=True)

    # additionally, neutralize false accusation to isolate the effect of that deception
    accused = truthified.get("accuse")
    if isinstance(accused, str) and accused and accused != "NONE":
        if _is_false_accusation(snapshot, accused):
            truthified["accuse"] = "NONE"

    # run meeting with override
    env.meeting(trigger="counterfactual", kill_event=kill_event, state_snapshot=snapshot, override_extracted={speaker: truthified})

    # continue play forward
    log = env.play()
    return asdict(log), log.winner or "killer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config used for original runs")
    ap.add_argument("--game_json", required=True, help="Path to a single game_XXXX.json to replay")
    ap.add_argument("--out", default="counterfactual_out", help="Output directory")
    ap.add_argument("--max_events", type=int, default=None, help="Max deceptive events to counterfactually test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    game = json.loads(Path(args.game_json).read_text(encoding="utf-8"))

    meetings = game.get("meeting_logs", []) or []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ites: List[dict] = []
    tested = 0
    max_events = args.max_events if args.max_events is not None else cfg.counterfactual_max_events_per_game

    for mi, m in enumerate(meetings):
        snap = m.get("state_snapshot_before_meeting") or m.get("state_snapshot")
        if not snap:
            continue

        kill_event = m.get("kill_event")
        for si, s in enumerate(m.get("statements", [])):
            labels = s.get("deception_labels", []) or []
            if not labels:
                continue
            if tested >= max_events:
                break

            speaker = s.get("speaker")
            stmt = s.get("extracted") or {}
            truth_map = s.get("truth") or {}

            cf_log, cf_winner = counterfactual_replay_for_event(
                cfg=cfg,
                snapshot=snap,
                kill_event=kill_event,
                speaker=speaker,
                original_stmt=stmt,
                truth_map=truth_map,
            )

            orig_winner = game.get("winner")
            # ITE in terms of innocent win probability (0/1)
            ite = (1.0 if cf_winner == "innocent" else 0.0) - (1.0 if orig_winner == "innocent" else 0.0)

            ites.append({
                "meeting_idx": mi,
                "statement_idx": si,
                "speaker": speaker,
                "labels": labels,
                "orig_winner": orig_winner,
                "cf_winner": cf_winner,
                "ite_innocent_win": ite,
            })

            # save full counterfactual log for auditing
            (out_dir / f"cf_m{mi:03d}_s{si:03d}.json").write_text(json.dumps(cf_log, ensure_ascii=False, indent=2), encoding="utf-8")

            tested += 1

        if tested >= max_events:
            break

    # Write ITE table
    ite_csv = out_dir / "ite.csv"
    with ite_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ites[0].keys()) if ites else ["meeting_idx","statement_idx","speaker","labels","orig_winner","cf_winner","ite_innocent_win"])
        w.writeheader()
        for row in ites:
            w.writerow(row)

    ate = sum(r["ite_innocent_win"] for r in ites) / max(1, len(ites))
    (out_dir / "ate.json").write_text(json.dumps({"ate_innocent_win": ate, "n": len(ites)}, indent=2), encoding="utf-8")

    print(json.dumps({"ate_innocent_win": ate, "n": len(ites), "out": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

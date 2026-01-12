from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _get_init_players(game: dict) -> List[dict]:
    for ev in game.get("events", []) or []:
        if ev.get("type") == "init" and ev.get("players") is not None:
            return ev.get("players") or []
    return []


def _killer_name(game: dict) -> Optional[str]:
    for p in _get_init_players(game):
        if p.get("role") == "killer":
            return p.get("name")
    return None


def compute_metrics(games: List[dict]) -> Dict[str, Any]:
    n = len(games)
    if n == 0:
        return {}

    winners = Counter([g.get("winner") for g in games])
    win_rate_innocent = winners.get("innocent", 0) / n
    win_rate_killer = winners.get("killer", 0) / n
    avg_turns = sum(int(g.get("turns", 0) or 0) for g in games) / n

    # Banishment metrics
    banish_tp = 0
    banish_fp = 0
    banish_fn = 0
    banish_events = 0

    # Deception metrics
    stmt_total = 0
    stmt_any_deception = 0
    deception_type_counts = Counter()
    deception_by_role = Counter()
    p_scores = []
    cred_scores = []

    # "Escape" metric: deceptive speaker not banished in that meeting
    deceptive_escape = 0
    deceptive_meetings = 0

    for g in games:
        killer = _killer_name(g)
        meetings = g.get("meeting_logs", []) or []

        for m in meetings:
            banished = m.get("banished")
            if banished:
                banish_events += 1
                if killer and banished == killer:
                    banish_tp += 1
                else:
                    banish_fp += 1

            # If the game ended by innocents winning, killer must have been banished at some point
            # FN counts games where banishment happened but killer was never banished.
            # (This is a coarse definition but stable.)
            # We'll compute FN per game below.

            # statements
            for s in m.get("statements", []) or []:
                stmt_total += 1
                labels = s.get("deception_labels", []) or []
                if labels:
                    stmt_any_deception += 1
                    for lb in labels:
                        deception_type_counts[lb] += 1
                    deception_by_role[s.get("role", "unknown")] += 1

                ps = s.get("p_score")
                if isinstance(ps, (int, float)):
                    p_scores.append(float(ps))
                cs = s.get("credibility")
                if isinstance(cs, (int, float)):
                    cred_scores.append(float(cs))

            # escape within meeting
            if banished:
                # if any speaker who was deceptive survived this meeting (wasn't banished), count it
                for s in m.get("statements", []) or []:
                    labels = s.get("deception_labels", []) or []
                    if labels:
                        deceptive_meetings += 1
                        if s.get("speaker") != banished:
                            deceptive_escape += 1
                        break  # count at most once per meeting

        # FN per game
        if banish_events > 0 and killer:
            killer_banished_in_game = any((m.get("banished") == killer) for m in meetings)
            if not killer_banished_in_game:
                banish_fn += 1

    precision = banish_tp / (banish_tp + banish_fp) if (banish_tp + banish_fp) > 0 else None
    recall = banish_tp / (banish_tp + banish_fn) if (banish_tp + banish_fn) > 0 else None

    deception_rate = stmt_any_deception / stmt_total if stmt_total > 0 else 0.0
    avg_p = sum(p_scores) / len(p_scores) if p_scores else None
    avg_cred = sum(cred_scores) / len(cred_scores) if cred_scores else None
    escape_rate = deceptive_escape / deceptive_meetings if deceptive_meetings > 0 else None

    return {
        "n_games": n,
        "win_rate_innocent": win_rate_innocent,
        "win_rate_killer": win_rate_killer,
        "avg_turns": avg_turns,
        "banishment_precision": precision,
        "banishment_recall": recall,
        "deception_rate_statements": deception_rate,
        "deception_type_counts": dict(deception_type_counts),
        "deception_by_role": dict(deception_by_role),
        "avg_p_score": avg_p,
        "avg_credibility": avg_cred,
        "deceptive_escape_rate_meeting": escape_rate,
    }

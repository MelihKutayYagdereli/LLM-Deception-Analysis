# project trial (V5)

A small social-deduction / deception sandbox inspired by the project paper requirements:
- 5-room house environment with hidden key + hallway escape
- LLM-driven agents (Option B)
- Structured meeting statements (schema v2) for **verifiable** deception labeling
- Credibility score `p` per claim (Gaussian oracle) + optional disclosure during voting
- Counterfactual evaluation: replace a deceptive statement with a truthful alternative and replay from the same snapshot

## Key design choices (V5)
- **Killer wins at 2 alive**: if only two players remain and the killer is alive, the killer wins (no artificial blocking).
- Meetings are triggered after kills by default (`meeting_after_kill_only: true`).
- In `discussion_mode: structured`, agents output a strict JSON statement and the environment renders it for all players.
  - This makes truth-checking and deception labels robust and reproducible.

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run experiments
```bash
python -m hoodwinked_lab.run_experiments --config configs/trial_gemini_v5.yaml --save_logs
```

Outputs a unique run directory:
- `config_resolved.json`
- `summary.json`
- `game_0000.json`, `game_0001.json`, ...

## Counterfactual replay (ITE / ATE)
Pick a saved game JSON from the experiment output:
```bash
python -m hoodwinked_lab.run_counterfactual \
  --config configs/trial_gemini_v5.yaml \
  --game_json runs/run_YYYYMMDD_HHMMSS/game_0000.json \
  --out counterfactual_out
```

This produces:
- `ite.csv` (one row per tested deceptive statement)
- `ate.json`
- `cf_mXXX_sYYY.json` (auditable counterfactual replays)

## Notes
- If you use `discussion_mode: free_text`, configure `judge_llm` to extract the structured schema from plain text.
- The mock provider is only for smoke tests; it does not play well.

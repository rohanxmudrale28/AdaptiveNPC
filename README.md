# Adaptive NPC: Player Behaviour Prediction and Counter-Strategy Generation

RL course mini project (B.Tech, Sem 7). Base paper: MBOM (Model-Based Opponent Modeling, NeurIPS 2022).

A small 1v1 Pygame arena game (9×9 grid, 8 actions: idle, 4 moves, attack, block, dodge) with an NPC that learns
a player's habits, predicts the next action, and counters it. It is a scaled-down reimplementation of ideas from
MBOM. Everything runs locally on CPU, with no external APIs.

The pieces, in build order:
- a GRU player-action predictor (finetuned online during a match)
- a PPO NPC conditioned on the predictor's output (`ppo_predicted`), against plain PPO (`ppo_none`) and a scripted NPC
- full MBOM: a learned env model Γ, level-0 and best-response opponent models (IOPs), recursive imagination, Bayesian mixing
- evaluation against scripted, held-out, online-learning and exploiter opponents (5 seeds, 95% CIs), plus a playable game with a demo panel

`PLAN.md` has the full design, the pre-registered criteria and every result. `outline.md` is the original spec.

## Setup (Windows / PowerShell)

Python 3.12. From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If script execution is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first. Tested with torch 2.14,
numpy 2.5, gymnasium 1.4, pygame 2.6, scipy 1.18, pytest 9.1. `requirements.txt` is unpinned.

Always run scripts from the repo root as modules (`python -m scripts.<name>`).

### Model checkpoints

Checkpoints (`*.pt`) are gitignored except the final deliverables in `models/final/` (predictor, env model,
`ppo_none` / `ppo_predicted` for seeds 0–4, the exploiters for seeds 0–2, and the pretrained naive / reasoning learners
for seeds 0–2 with the reasoning learner's NPC model; about 1.7 MB). The configs read from `models/`, so after a fresh clone, install them:

```powershell
Copy-Item models\final\predictor.pt, models\final\env_model.pt models\
New-Item -ItemType Directory -Force models\npc | Out-Null
Copy-Item models\final\npc\*.pt models\npc\
New-Item -ItemType Directory -Force models\exploiter, models\learners | Out-Null
Copy-Item models\final\exploiter\*.pt models\exploiter\
Copy-Item models\final\learners\*.pt models\learners\
```

This is enough to play, run the tests, and run the evaluations, including the exploiter and learner cells.
The retraining commands below are only needed to reproduce the checkpoints themselves.

### Check the install

```powershell
python -m pytest -q          # headless, about 1 minute, 156 tests
```

## Play the game

```powershell
python -m scripts.play                                   # title screen, then opponent select
python -m scripts.play --panel on                        # same, demo panel visible from the start
python -m scripts.play --opponent rusher --rounds 3      # straight into a match (skips the menu)
python -m scripts.play --opponent mbom --panel on        # MBOM enemy ("ECHO") with the demo panel
python -m scripts.play --opponent mbom --blind --label B --participant P01
python -m scripts.play --tick-rate 8                     # faster game ticks (default 4 steps/s)
```

Menu enemies: TRAINING DUMMY (camper), RUSTY (scripted NPC), REFLEX (`ppo_none`), SEER (`ppo_predicted`),
ECHO (MBOM). `--opponent` also accepts any archetype (`rusher`, `camper`, `repeater`, `mixer`, `flanker`,
`switcher`, `kiter`, `counter_puncher`) or `random`. Settings are in `configs/render.yaml` and `configs/ui.yaml`
(`--set ui.<key>=value` overrides `ui.yaml`).

Controls: WASD / arrows move, **left click** attack, **hold right click** block, **Q** dodge, Enter confirm, Esc pause,
H legend, F scanlines (mouse clicks only count during active play; menus are keyboard-only). `--keys keyboard`
restores the old all-keyboard set (**J** attack, **K** block, **L** dodge; bindings in `configs/keys_keyboard.yaml`).
Also:
**Tab** toggles the demo panel, **P** saves a screenshot to `results/demo/`, **M** mutes / unmutes all sound,
**+** / **-** change the volume (default 0.5, set in `configs/ui.yaml`). Sound is synthesized in code; `--mute` starts
silent and `--no-music` keeps the effects only. With `--blind` (and in `scripts/human_eval.py`) music is off by
default (`--music` turns it on) and every opponent gets the same effects.

- **Demo panel (Tab, or `--panel on`):** live bars for the NPC's predicted next player action (the action you took is
  highlighted), a running prediction accuracy, and, against the MBOM enemy, the mixer weights α over the IOP levels
  with the 0.79 cap marked.
- **`--blind`:** hides enemy names, the NPC header and the panel, and logs the real NPC version to `results/human/`
  (one JSON per match, participant ID only). Use it for blind human tests.
- The MBOM enemy approximates the evaluated MBOM: its imagination update runs in a worker thread, so it lands a few
  steps late. Set `--set ui.mbom_async=false` for the synchronous version used in evaluation (the enemy then stalls during each update).

## Reproduce the results

Smoke configs (`configs/experiments/*_smoke.yaml`) run each pipeline end to end in about a minute. The full runs
below take hours each (training and the final grid) and write to `results/`. Every run copies its resolved config and
git hash next to its outputs, and every seed lives in a config file.

**1. Predictor** (GRU, `results/predictor/main`)

```powershell
python -m scripts.train_predictor --config configs/predictor.yaml    # collects data first if missing
python -m scripts.eval_predictor  --config configs/predictor.yaml    # accuracy / KL table, recovery curve
python -m scripts.sweep_online    --config configs/predictor.yaml    # online-finetune settings sweep
```

**2. NPC** (PPO, `results/npc/main`; 3 conditioning modes × seeds 0–4)

```powershell
python -m scripts.train_npc --config configs/npc_ppo.yaml            # or --mode predicted --seed 0
python -m scripts.eval_npc  --config configs/npc_ppo.yaml
```

**3. Env model Γ and MBOM** (`results/env_model/main`, `results/mbom/*`)

```powershell
python -m scripts.train_env_model --config configs/env_model.yaml    # collects transitions first (needs models/learners, models/exploiter, installed above)
python -m scripts.eval_mbom       --config configs/mbom.yaml
python -m scripts.probe_cond_shift --config configs/mbom.yaml        # conditioning-shift probe
python -m scripts.benchmark_mbom  --config configs/mbom.yaml         # speed
python -m scripts.gonogo_mbom     --config configs/mbom.yaml         # go/no-go on the development exploiter
python -m scripts.dev_variants    --config configs/mbom.yaml         # dev variants; add --adaptive, or --test h2_test
```

**Learners and exploiters** (opponents of the MBOM and final-grid cells; the checkpoints are in `models/final/`, so
skip the training commands unless you want to regenerate them)

```powershell
python -m scripts.train_predictor --config configs/learners_npc_model.yaml
python -m scripts.train_npc       --config configs/learners_pretrain.yaml
python -m scripts.check_learner_drift --config configs/learners.yaml
python -m scripts.train_exploiter --config configs/exploiter.yaml    # 1M steps per seed
python -m scripts.eval_exploiters --config configs/exploiter.yaml    # cross-evaluation
```

**4. Final grid** (`results/m10/final`; pre-registered in `PLAN.md`, M10, with Amendment 1)

```powershell
python -m scripts.final_grid --config configs/mbom.yaml --threads 1                  # whole grid, resumable
python -m scripts.final_grid --config configs/mbom.yaml --methods mbom --seeds 0 --threads 1   # one part, run several terminals in parallel
python -m scripts.final_grid --config configs/mbom.yaml --aggregate-only             # methods.csv, comparisons.csv
```

The full grid is about 49 process-hours (about 5 h on 10 terminals). The exploiter cells need `models/exploiter/` (installed above).
`final_grid` refuses `exploiter_seed1`, the development exploiter.

**5. Plots**

```powershell
python -m scripts.make_plots --config configs/mbom.yaml              # results/m11/: claim groups, early/late, α traces, tables
python -m scripts.plot_adaptive_rounds --config configs/mbom.yaml    # results/mbom/adaptive_test/win_rate_by_round.png
python -m scripts.balance_matrix --config configs/balance.yaml       # results/balance/ heatmaps
python -m scripts.eval_predictor --config configs/predictor.yaml     # recovery curve
```

`make_plots` and `plot_adaptive_rounds` only read saved results, so they are quick and rerun without any new
matches. The figures used in the report are copied to `report/figures/`, with `report/figures/captions.md` giving
the source file of each.

## Results summary

Win rates are from the NPC's side, draws count as ½, intervals are 95% CIs. Full tables are in `PLAN.md`.

**Predictor.** The GRU predicts the next action of the train archetypes well, from 0.73 to 0.96 top-1 against the scripted NPC (rusher
0.91, repeater 0.96, flanker 0.91, switcher 0.88, camper 0.73). The mixer is at its ceiling of 0.25 (chance 0.125).
Held-out archetypes are harder: kiter 0.86, counter-puncher 0.47 frozen and 0.61 with online finetuning.
(`results/predictor/main/`)

**NPC.** Both PPO variants beat the scripted NPC. Against the fixed archetypes, `ppo_none`, `ppo_predicted` and
the oracle-conditioned upper bound are all near 100%, so win rate can't separate them there.
(`results/npc/main/`)

**Env model Γ.** Full-state exactness is 99.4% (1 step) and 98.8% (2 steps) on-policy, and 98.9% / 97.3% under random actions.
(`results/env_model/main/`)

**Exploiters and learners.** Online learners beat the scripted NPC and are level with `ppo_none`, but lose to
`ppo_predicted` within a 30-round match. Offline exploiters trained for 1M steps do beat `ppo_predicted` (0.70–0.81
mean). The exploit transfers to NPC checkpoints they never saw (seeds 1–4) but unevenly (about 0.61 on unseen
`ppo_predicted` cells, range 0.07–0.995), and on those seeds it hurts `ppo_none` about as much as `ppo_predicted`.

**MBOM (negative findings).** MBOM did not beat its baselines:
- **Go/no-go (development exploiter): NO-GO.** MBOM − `ppo_predicted` = +0.004 (needed ≥ 0.05). MBOM trails `mbom_no_imagination` by 0.050 ± 0.031.
- **Adaptive opponents (H1): not supported.** Against the naive and reasoning learners, MBOM − `no_imagination` = −0.011 ± 0.020.
- **Lookahead opponent (H2): not supported.** MBOM − `no_imagination` = −0.007 ± 0.041. All methods won only about 0.10–0.12 of rounds (floor effect).
- **Final grid** (exploiter claim groups, NPC seeds 0–4, 7 matches per cell):
  - MBOM vs `ppo_predicted`: no detectable difference in any claim group.
  - MBOM vs `no_imagination`: MBOM is **worse** on seeds 1–4, pooled over both exploiters (−0.045) and for `exploiter_seed0` (−0.052).
  - The Bayesian mixer beats uniform mixing (`uniform_mix` − MBOM = −0.055 pooled on seeds 1–4).
  - No method separates at NPC seed 0.
- **Why:** the imagined best-response levels (φ1, φ2) predict these opponents worse than the pretrained φ0 (around chance, 0.10–0.21 accuracy vs 0.75–0.86). The mixer down-weights them but its α cap leaves them 22% of the weight, which blurs the policy input. Imagination with the true simulator behaves the same, so the cause isn't Γ's error. The per-level and mixer evidence is in `PLAN.md` (M8).
- The single-level and `top_level` variants are development-set results only (`results/mbom/dev_variants.csv`).

**Practical gain over the baselines.** What helped was the φ0 online finetune: `mbom_no_imagination` beats `ppo_predicted` by +0.035 pooled on seeds 1–4 (Holm p = 0.011, below the 0.05 practical threshold).

## Known limitations

Full list in `PLAN.md`. The main ones:
- **Saturation:** win rate is about 100% against the fixed archetypes for every learned NPC, so those cells can't rank methods.
- **Matches are 30 rounds** (a default, not tuned). The learners get roughly 600–1,000 steps, about 0.1% of the exploiters' training budget, so the adaptive-opponent results describe early adaptation. The NPC's win rate against both learners is still falling at round 30.
- **Matches in a cell share one NPC checkpoint**, so paired tests treat them as independent and understate the uncertainty. The 15 cross-eval cells are not independent either. Per-seed results are in the CSVs.
- **Exploiters were trained against NPC seed 0**, so seed 0 and seeds 1–4 are reported separately. Only 5 NPC checkpoints exist per method.
- **Γ errors compound with k.** k > 2 was not measured. The weakest cell is the reward on KO-loss steps (about 86%).
- **The demo MBOM enemy only approximates the evaluated MBOM** (asynchronous update).
- **Human evaluation (M13) was not run.** The `--blind` mode logs matches, but there is no analysis script or human-study result.
- **Archetype balance:** the flanker is weak overall, the passive archetypes mostly draw, and `counter_puncher` still wins 90%+ against deterministic opponents (it isn't dominant).
- **KL is uninformative for near-deterministic learner policies:** the drift threshold passes trivially, so score, damage and entropy carry the information.
- **`ppo_none`, `ppo_predicted` and the MBOM variants use PPO checkpoints saved as plain torch `.pt`:** SB3's zip load fails with the installed torch 2.14.

## Layout

```
adaptive_npc/   package: env (pygame-free core), players, models, npc (+ mbom/), eval, render, ui
configs/        every seed and hyperparameter (YAML); configs/experiments/ has the smoke configs
scripts/        CLI entry points (python -m scripts.<name> --config ...)
tests/          pytest suite (headless)
models/final/   final deliverable checkpoints (everything else in models/ is gitignored)
results/        committed metrics, config snapshots and plots per experiment
report/figures/ final figures and captions.md
```

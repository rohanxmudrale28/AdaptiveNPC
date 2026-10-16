# Adaptive NPC — Build Plan

Scaled-down MBOM (Yu et al., NeurIPS 2022) in a custom 1v1 Pygame arena. The pieces, in build order:
- GRU player-action predictor
- prediction-conditioned PPO NPC
- full MBOM: env model, level-0 IOP, recursive imagination with k-step rollouts, Bayesian mixing
- evaluation against fixed, naive-learner and reasoning-learner opponents (5 seeds, 95% CI), human play, and a demo panel

The full spec is in [outline.md](outline.md) and the working rules are in [CLAUDE.md](CLAUDE.md).

Workflow: one milestone at a time, in the order build → test → commit. Long training runs are run by the user. I verify each script with a smoke config instead.

---

## Paper notes (MBOM) used in this plan
The PDF in `references/` has no appendices, so the Appendix E hyperparameters and the code link are missing.
- **Env model** Γ(s, a, aᵒ) → (s′, r), trained with MSE loss (eq. 2).
- **Level-0 IOP** φ0, trained by maximum likelihood on (s, aᵒ) (eq. 3).
- **Opponent best response:** for each opponent action, uniformly sample k-step continuations in Γ. Score each with the opponent reward rᵒ plus γ^{k+1}·Vᵒ, where Vᵒ = −V because the game is zero-sum (eq. 5). During these rollouts the agent acts with π(·|s, ã ~ IOP_{m−1}). Finetune φ_{m−1} on the best responses to get φ_m.
- **Bayesian mixing:**
  - posterior p(m|aᵒ) ∝ π̃_m(aᵒ|s)·p(m), with the prior = moving average of past posteriors
  - Ψ = decayed average of the posteriors
  - α = softer-softmax(Ψ) (eq. 6–8)
- **What adapts online:** the NPC policy θ is pretrained with PPO and frozen during interaction. Only the IOPs and α change.
- **Ablations:** w/o IOPs (φ0 only), MBOM-BM (finetune on random actions), single φ_m, uniform mixing.

---

## File structure
```
AdaptiveNPC/
  configs/                      # ALL seeds / hyperparams (YAML)
    env.yaml                    # arena size, HP, stamina, cooldowns, reward weights, max steps
    players.yaml                # archetype params, train vs held-out split, switch schedule
    balance.yaml                # round-robin rounds per pair, dominance threshold
    render.yaml                 # human-play tick rate, fps, tile size
    predictor.yaml              # GRU sizes, window, lr, online finetune settings
    npc_ppo.yaml                # SB3 PPO hparams, n_envs, vec env type, conditioning mode, opponent pool
    env_model.yaml
    mbom.yaml                   # M levels, k, rollout samples, mixer decay/temperature
    learners.yaml               # naive / reasoning learner settings, lr, KL-change threshold
    experiments/{main,ablations,generalization,adaptation,smoke}.yaml
  adaptive_npc/                 # importable package (run scripts from repo root with python -m)
    config.py  seeding.py  logging_utils.py
    env/
      actions.py                # Action enum, constants
      core.py                   # ArenaGame: pure numpy rules, simultaneous step(a_npc, a_ply). No pygame.
      observations.py           # encode state from NPC view / player view (mirror)
      gym_env.py                # NPCEnv(gym.Env) wraps core + embedded opponent; PlayerEnv mirror
      vec_env.py                # make_vec_env(): seeded DummyVecEnv / SubprocVecEnv from config
      match.py                  # multi-round match runner, persistent opponent/NPC state, hooks
    players/
      base.py                   # Player interface: reset(), act(obs, hist), observe(transition)
      scripted.py               # rusher, camper, repeater, mixer, switcher, held-out archetypes
      naive_learner.py          # PPO-updating player
      reasoning_learner.py      # models NPC, policy conditioned on predicted NPC action
      human.py                  # keyboard player
    models/
      player_model.py           # GRU next-action predictor (also used as IOP φ0)
      env_model.py              # Γ: MLP (s, a, aᵒ) → (Δs, r)
      ppo_lite.py               # minimal torch PPO for ONLINE updates of learning players
    npc/
      scripted_npc.py           # hand-coded baseline NPC
      ppo_npc.py                # SB3 PPO + obs wrapper (none | predicted dist | oracle)
      mbom/
        iops.py                 # IOP set φ0..φM-1 (copies of predictor), finetune
        imagination.py          # batched k-step rollouts in Γ → opponent best responses
        mixer.py                # Bayesian mixing + softer-softmax
        agent.py                # MBOMAgent: act/observe/online update loop (Alg. 1)
    eval/
      metrics.py                # accuracy, KL, win rate, return, adaptation time, mean ± 95% CI
      runner.py                 # method × opponent × seed grid → results/
      plots.py
    render/
      view.py                   # Pygame drawing of arena + HUD (reads core state only)
      demo_panel.py             # predicted next-action bars, belief weights α / player-type
      input.py                  # keyboard → Action
  scripts/                      # thin CLI entry points: --config, --seed, --out
    benchmark_env.py  balance_matrix.py  collect_data.py  train_predictor.py  train_npc.py
    train_env_model.py  train_learners.py  check_learner_drift.py  run_experiment.py
    make_plots.py  play.py  human_eval.py
  tests/                        # pytest: env rules, determinism, vec env, players, predictor, mixer, imagination
  data/        (gitignored)     # collected trajectories (.npz)
  models/      (gitignored)     # checkpoints; final deliverable models whitelisted in models/final/
  results/                      # committed: metrics CSV, config snapshot, plots per experiment/seed
  report/                       # figures + notes for the report
```
Conventions:
- The core has no pygame import, so training runs headless.
- Every script reads its YAML config.
- Every script copies the resolved config and the git hash into `results/<exp>/<method>/<opponent>/seed_<n>/`.

---

## Milestones

### Tier 0 — Setup
- [x] **M0 Bootstrap**
  - [x] Rename branch `master` → `main`
  - [x] Make the initial commit of the existing docs
  - [x] scipy checked with `pip show`: 1.17.1 is in system Python, but it was missing from `.venv` and requirements.txt, so it was added to requirements.txt
  - [x] Add `pytest` to requirements.txt
  - [x] Install deps in `.venv`
  - [x] `config.py` (YAML load + CLI overrides) and `seeding.py` (python/numpy/torch/gym)
  - [x] Package skeleton and pytest config
  - [x] Check: `python -c "import adaptive_npc"` works and `pytest` runs
  - [x] Commit

### Tier 1 — Playable game + baselines (MVP)
- [x] **M1 Env core**
  - [x] `actions.py`, `core.py`, `observations.py`, `gym_env.py` (NPCEnv embeds any Player), `match.py`
  - [x] `vec_env.py`: `make_vec_env()` builds DummyVecEnv or SubprocVecEnv from config (`n_envs`, `vec_env_type`). Each sub-env gets a distinct, reproducible seed (`base_seed + i`).
  - [x] Tests for each rule: hit, block/parry, dodge i-frames, stamina, cooldown, timeout
  - [x] Test: identical trajectories with the same seed
  - [x] Test: vectorized results are reproducible, and sub-envs differ from each other
  - [x] Test: gymnasium `check_env` passes
  - [x] `benchmark_env.py`: steps/s for random vs random in three setups:
    - [x] raw core
    - [x] single gym env
    - [x] vectorized: Dummy and Subproc with `n_envs` ∈ {1, 4, 8}
  - [x] Save benchmark results to `results/benchmarks/env_speed.csv`
  - [x] Target: at least 10k steps/s for a single headless env
  - [x] Commit
- [ ] **M2 Scripted players**
  - [ ] Train archetypes: rusher, camper, pattern repeater, random mixer, switcher
  - [ ] Held-out archetypes: kiter, counter-puncher
  - [ ] Scripted NPC baseline
  - [ ] Archetypes can play either side, using the mirrored player-view observation
  - [ ] Test: each archetype's action histogram differs from the others
  - [ ] Test: the switcher switches on schedule
  - [ ] Test: the scripted NPC beats random by more than 80%
  - [ ] Commit
- [ ] **M3 Game balance**
  - [ ] `balance_matrix.py`: round-robin over all archetypes (train + held-out + scripted NPC + random)
    - [ ] 200 rounds per pair, seeds [0, 1, 2], both side assignments
  - [ ] Output: win-rate matrix CSV and heatmap in `results/balance/`, draws counted as ½
  - [ ] Check: mirror matches are about 50% (45–55%). This confirms the game has no side bias.
  - [ ] Check: no dominant archetype. An archetype is dominant if it wins at least 65% against every other archetype. The threshold is set in `balance.yaml`.
  - [ ] Check: every non-random archetype has at least one matchup it wins and at least one it loses
  - [ ] If an archetype dominates, adjust the params in `configs/env.yaml` and re-run until the checks pass. Candidate params: damage, cooldowns, stamina costs/regen, parry stun, dodge distance. Never hardcode params in scripts.
  - [ ] Log each tuning iteration (params → matrix) to `results/balance/log.md`
  - [ ] Commit the final balanced `env.yaml` with its matrix
- [ ] **M4 Rendering + human play**
  - [ ] `render/view.py`, `render/input.py`
  - [ ] `scripts/play.py` (human vs random/scripted)
  - [ ] Tick rate (game steps/s) is configurable through `render.yaml` and the `--tick-rate` CLI override
  - [ ] Render fps is independent of the game tick rate
  - [ ] Render reads state only. The core stays pygame-free.
  - [ ] Check: a round can be played by keyboard at the default and at a changed tick rate
  - [ ] Commit
- [ ] **M5 Player predictor (GRU)**
  - [ ] `collect_data.py`: windowed (state, npc action, player action) histories vs the train archetypes
  - [ ] `train_predictor.py`: offline training
  - [ ] Online finetuning during a match
  - [ ] Report top-1 accuracy and KL per archetype, compared with majority-action and last-action baselines
  - [ ] Report the accuracy recovery curve after a style switch
  - [ ] Commit
- [ ] **M6 PPO NPC + eval harness**
  - [ ] `ppo_npc.py` with 3 conditioning modes:
    - [ ] `none` (plain PPO)
    - [ ] `predicted` (GRU output appended to the observation)
    - [ ] `oracle` (debug upper bound)
  - [ ] `train_npc.py` vs the train-archetype pool, using `make_vec_env`
  - [ ] `eval/metrics.py` and `eval/runner.py`
  - [ ] Smoke run writes `results/.../metrics.csv`
  - [ ] Full runs (user): both PPO variants beat the scripted NPC vs the train archetypes
  - [ ] Commit

### Tier 2 — Core contribution
- [ ] **M7 Env model Γ**
  - [ ] `env_model.py` and `train_env_model.py`, trained on the buffer from M6 rollouts
  - [ ] Report one-step and k-step state/reward error
  - [ ] Commit
- [ ] **M8 MBOM**
  - [ ] `iops.py`: φ0 is the pretrained predictor
  - [ ] `imagination.py`: eq. (5), batched in torch
  - [ ] `mixer.py`: eq. (6–8)
  - [ ] `agent.py`: online loop (Alg. 1). The policy is the frozen `predicted` PPO, fed the mixed IOP.
  - [ ] Ablation flags: `no_imagination`, `random_finetune` (BM), `single_level=m`, `uniform_mix`
  - [ ] Test: the mixer converges to the generating level on synthetic data
  - [ ] Test: imagination finds the known best response in a hand-built state
  - [ ] Commit
- [ ] **M9 Learning opponents**
  - [ ] `ppo_lite.py`
  - [ ] Naive learner: pretrained vs the scripted NPC, then updated online every N steps
  - [ ] Reasoning learner: its own GRU model of the NPC. Its policy is conditioned on the predicted NPC action, and both parts are finetuned online.
  - [ ] `check_learner_drift.py`: measures whether each learner's policy actually changes during a 30-round match against a frozen NPC
    - [ ] Fix a probe set of 512 states from the match's first rounds
    - [ ] Snapshot the policy at match start and match end
    - [ ] Report mean KL(π_start ‖ π_end) over the probe states, saved to `results/learners/drift.csv`
  - [ ] Check: mean KL ≥ `kl_min` (default 0.05 nats, in `learners.yaml`) for both learners
  - [ ] If the check fails, tune the online LR (and update frequency) in `learners.yaml` until KL ≥ `kl_min`. The LR must not get so high that the learner collapses: it must still beat random and must not lose more than at match start.
  - [ ] Log the chosen LR and its KL per seed
  - [ ] Test: the naive learner's win rate rises over a match vs a frozen NPC
  - [ ] Test: the reasoning learner's NPC-prediction accuracy is above chance
  - [ ] (Optional) Retrain the NPC with learner snapshots in the opponent pool
  - [ ] Commit

### Tier 3 — Experiments
- [ ] **M10 Experiment grid**
  - [ ] Methods: scripted, PPO, pred-PPO, MBOM, and 4 ablations
  - [ ] Opponents: fixed archetypes, switcher, naive, reasoning, held-out
  - [ ] Seeds: [0, 1, 2, 3, 4]
  - [ ] Metrics: win rate, return, prediction accuracy/KL, adaptation time after a switch
  - [ ] Commit configs and results
- [ ] **M11 Plots + tables**
  - [ ] Bar charts with 95% CI
  - [ ] Adaptation curves: win rate and prediction error vs rounds since the switch
  - [ ] α-weight traces
  - [ ] Generalization table (held-out archetypes)
  - [ ] Commit

### Tier 4 — Demo + human study
- [ ] **M12 Demo panel**
  - [ ] Live bars for the predicted next action
  - [ ] α over IOP levels
  - [ ] Posterior over player archetypes, shown in `play.py`
  - [ ] Commit
- [ ] **M13 Human eval**
  - [ ] `human_eval.py`: repeated matches per participant, logging to `results/human/`, with the tick rate taken from `render.yaml`
  - [ ] Analysis script
  - [ ] Commit
- [ ] **M14 Wrap-up**
  - [ ] README usage section
  - [ ] Final models in `models/final/`
  - [ ] Report figures
  - [ ] Commit

---

## Decisions (approved defaults)
1. **Arena:** discrete grid 9×9, rendered at 64 px/tile.
2. **Action set (8):** `idle, up, down, left, right, attack, block, dodge`.
   - Moving sets the facing direction.
   - Attack hits the adjacent tile in the facing direction.
   - Block protects the front only.
   - Dodge dashes 2 tiles with i-frames.
3. **Trade-offs:** these are the starting values, which may be retuned in M3.
   - HP 5 each.
   - Attack: 1 damage, cooldown 2 steps.
   - Block: negates a frontal hit and stuns the attacker for 2 steps (parry). Costs 1 stamina per step.
   - Dodge costs 3 stamina.
   - Stamina max 6, regenerates 1 every 2 steps.

   Resulting triangle: attack > move, block > attack, flank/wait > block.
4. **Turn model:** both players act simultaneously. Actions resolve in a fixed order: dodge → move → block → attack.
5. **Rounds and matches:**
   - A round ends when HP reaches 0 or after 150 steps. At timeout, higher HP wins, otherwise it's a draw.
   - A match is 30 rounds. Opponent models persist across rounds within a match.
   - The switcher changes archetype every 10 rounds.
   - A within-round variant switches at step 75.
6. **Reward (zero-sum, rᵒ = −r):** ±1 for win/loss, 0 for a draw. Shaping: +0.1 per damage dealt, −0.1 per damage taken.
7. **Observation:**
   - NPC-centric vector (~20 dims): relative position, facing, HP, stamina, cooldowns and stun for both players, plus time fraction.
   - The predictor sees a window of the last H = 8 steps of (obs, npc action, player action).
8. **Archetypes:**
   - Train: rusher, camper, pattern repeater, random mixer, switcher (cycles through the train archetypes).
   - Held-out: kiter (hit and retreat), counter-puncher (blocks until attacked, then punishes).
9. **Network sizes:**
   - GRU: 1 layer, 64 hidden units, plus a linear head.
   - PPO: MLP [64, 64] for both policy and value.
   - Env model: MLP [256, 256], predicting Δs and r.
   - IOPs use the same GRU architecture as the predictor.
10. **MBOM settings:**
    - M = 3 levels (φ0, φ1, φ2).
    - Rollout horizon k = 2, with the Vᵒ = −V bootstrap.
    - 16 sampled continuations per opponent action.
    - Finetune the IOPs every 50 real steps on the last 256 states: 3 gradient steps, lr 1e-3.
    - Mixer: decay 0.9, α = softmax(Ψ / T) with T = 0.5. T is tuned on train opponents only.
11. **Env model for imagination:** the learned Γ, as in the paper. The true simulator is available behind a debug-only `oracle_model` flag and is never used in reported results.
12. **PPO tooling:**
    - Offline NPC training uses SB3 PPO with vectorized envs. Default: `n_envs` = 8, `DummyVecEnv`. Switch to `SubprocVecEnv` if the M1 benchmark shows it is faster.
    - Online updates of learning players use a small in-house `ppo_lite.py` (~150 lines). SB3 can't easily step a policy inside another loop.
13. **Training budget (CPU):**
    - NPC PPO: 2M steps per seed, about 30–60 min each. The user runs these.
    - Predictor data: 200 rounds per archetype.
    - Smoke configs take about 1 min. I run these to verify each script.
14. **Seeds and statistics:** seeds `[0, 1, 2, 3, 4]` are set in the experiment configs. 95% CI is a t-interval computed with scipy.
15. **Model artifacts:** checkpoints stay out of git (`*.pt` and `*.zip` are gitignored). Only the final deliverable models are whitelisted, in `models/final/`.
16. **Dependencies:** `scipy` has been added to requirements.txt. `pytest` will be added in M0. Nothing else.
17. **Git:** rename `master` → `main` before the first commit.
18. **Human eval:**
    - 5–8 participants, 3 matches × 10 rounds each vs MBOM.
    - Log predictions, α and outcomes.
    - No personal data beyond a participant ID.
19. **Game balance:**
    - 200 rounds per pair × 3 seeds.
    - An archetype is dominant if it wins at least 65% against every other archetype.
    - Mirror-match tolerance: 45–55%.
20. **Human-play tick rate:**
    - Default: 8 game steps/s, with rendering at 60 fps.
    - Set in `render.yaml`, overridable with `--tick-rate`.
21. **Learner drift:**
    - KL threshold `kl_min` = 0.05 nats, measured on 512 probe states between match start and match end.
    - Starting online LR: 3e-4. Tune upward until the threshold is met.

---

## Verification (every milestone)
- `pytest -q` passes, with new tests added each milestone.
- `python -m scripts.benchmark_env --config configs/env.yaml` reports steps/s for the single and vectorized envs.
- `python -m scripts.balance_matrix --config configs/balance.yaml` writes the matrix and the pass/fail result of the dominance check.
- `python -m scripts.check_learner_drift --config configs/learners.yaml` reports the start→end KL per learner.
- Each training script has a smoke config (`configs/experiments/smoke.yaml`). I run it end to end and check that `results/.../metrics.csv` is written. The user runs the full-budget runs and pastes summaries.
- `python -m scripts.play --tick-rate 8` for manual checks of rendering, controls and the demo panel.
- Commit after each green milestone, for example `M3: game balance matrix + tuned env params`.

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
      scripted.py               # rusher, camper, repeater, mixer, flanker, switcher, held-out archetypes
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
      clock.py                  # fixed-timestep TickClock: game tick rate independent of render fps
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
  - [x] `eval_predictor.py`: mixer entropy ceiling (max weight 0.25), recovery curve, online vs frozen
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
  - [x] `eval_predictor.py`: mixer entropy ceiling (max weight 0.25), recovery curve, online vs frozen
  - [x] Commit
- [x] **M2 Scripted players**
  - [x] Train archetypes: rusher, camper, pattern repeater, random mixer, switcher (flanker added in M3)
  - [x] Held-out archetypes: kiter, counter-puncher
  - [x] Scripted NPC baseline
  - [x] Archetypes can play either side, using the mirrored player-view observation
  - [x] Test: each archetype's action histogram differs from the others
  - [x] Test: the switcher switches on schedule
  - [x] Test: the scripted NPC beats random by more than 80%
  - [x] `eval_predictor.py`: mixer entropy ceiling (max weight 0.25), recovery curve, online vs frozen
  - [x] Commit
- [x] **M3 Game balance**
  - [x] `balance_matrix.py`: round-robin over all archetypes (train + held-out + scripted NPC + random)
    - [x] 200 rounds per pair, seeds [0, 1, 2], both side assignments (200 rounds per seed per side)
  - [x] Output: win-rate matrix CSV and heatmap in `results/balance/`, draws counted as ½. Also written: decisive win-rate matrix (wins / (wins + losses)) and draw-rate matrix.
  - [x] Report the draw rate per pair and flag pairs above 50% draws (`draw_flag_threshold`; warning only)
  - [x] Check: mirror matches are about 50% (45–55%), measured on the NPC-slot win rate. This confirms the game has no side bias.
  - [x] Check: no dominant archetype. An archetype is dominant if it wins at least 65% against every other archetype, on either the draw-adjusted or the decisive win rate. The threshold is set in `balance.yaml`.
  - [x] Check: every non-random archetype has at least one matchup it wins and at least one it loses
  - [x] Check: every non-random archetype has at least one counter that beats it by 60% or more, on both the draw-adjusted and the decisive win rate (`counter_threshold`)
  - [x] If an archetype dominates, adjust the params and re-run until the checks pass. `counter_puncher` dominated, and `scripts/diagnose_counter_puncher.py` showed the rules allow a counter (flanking beats block). The fix was therefore in the players, not in `configs/env.yaml`: `flanker` added as a train archetype, and `counter_puncher` softened to `reaction_delay: 1`, `noise: 0.2` (in `players.yaml`).
  - [x] Log each tuning iteration (params → matrix) to `results/balance/log.md`
  - [x] Commit the final balanced config with its matrix (`env.yaml` is unchanged)
- [x] **M4 Rendering + human play**
  - [x] `render/view.py`, `render/input.py`, `render/clock.py`
  - [x] `scripts/play.py` (human vs random/scripted)
  - [x] Tick rate (game steps/s) is configurable through `render.yaml` and the `--tick-rate` CLI override
  - [x] Render fps is independent of the game tick rate
  - [x] Render reads state only. The core stays pygame-free.
  - [x] Check: a round can be played by keyboard at the default and at a changed tick rate
  - [x] `eval_predictor.py`: mixer entropy ceiling (max weight 0.25), recovery curve, online vs frozen
  - [x] Commit
- [x] **M5 Player predictor (GRU)**
  - [x] `collect_data.py`: windowed (state, npc action, player action) histories vs the train archetypes, with both the scripted and a random NPC
  - [x] `train_predictor.py`: offline training
  - [x] Online finetuning during a match
  - [x] Report top-1 accuracy and KL per archetype, compared with majority-action and last-action baselines
  - [x] Report the accuracy recovery curve after a style switch
  - [x] `eval_predictor.py`: mixer entropy ceiling (max weight 0.25), recovery curve, online vs frozen
  - [x] Commit
- [x] **M6 PPO NPC + eval harness**
  - [x] `ppo_npc.py` with 3 conditioning modes:
    - [x] `none` (plain PPO)
    - [x] `predicted` (GRU output appended to the observation)
    - [x] `oracle` (debug upper bound)
  - [x] `train_npc.py` vs the train-archetype pool, using `make_vec_env`
  - [x] `eval/metrics.py` and `eval/runner.py` (`scripts/eval_npc.py` is the CLI; PPO checkpoints are plain torch `.pt` because SB3's zip load fails with the installed torch 2.14)
  - [x] Metrics also report episode length (`mean_steps`), `damage_taken` per round, and `win_rate_first_n` (N = `eval.first_n_rounds`)
  - [x] Smoke run writes `results/.../metrics.csv`
  - [x] Full runs (user): both PPO variants beat the scripted NPC vs the train archetypes
  - [x] Commit

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
- [x] **M9 Learning opponents** (built before M7/M8)
  - [x] `ppo_lite.py`: trains the SB3 `ActorCriticPolicy` loaded from a `.pt` in place (GAE, clip, normalized advantages, `target_kl` early stop, grad clip, `critic_warmup_updates`)
  - [x] Naive learner: pretrained vs the scripted NPC, then updated online every N steps
  - [x] Reasoning learner: its own GRU model of the NPC. Its policy is conditioned on the predicted NPC action, and both parts are finetuned online.
  - [x] Pretraining reuses existing scripts with configs only (no `train_learners.py`), because observations are mirror-symmetric:
    - NPC model: `train_predictor --config configs/learners_npc_model.yaml` (labels are scripted-NPC actions)
    - Policies: `train_npc --config configs/learners_pretrain.yaml` (`none` → naive, `predicted` → reasoning, vs `[scripted_npc]`)
  - [x] `check_learner_drift.py`: measures whether each learner's policy actually changes during a 30-round match against a frozen NPC
    - [x] Frozen NPCs: `scripted`, `ppo_none` and `ppo_predicted` (easy / medium / hard; seed-0 checkpoints, player predictor frozen)
    - [x] Fix a probe set of 512 states, from a separate warm-up rollout of the frozen start learner
    - [x] Snapshot the policy at match start and match end; KL and policy entropy are also logged after every round
    - [x] Report mean KL(π_start ‖ π_end) over the probe states, saved to `results/learners/drift.csv` (`kl_full`; `kl_policy` and `npc_model_kl` split the reasoning learner's change by part)
    - [x] Round-by-round outcome per opponent against a frozen-learner control, with a beat/match/lose verdict from the 95% CI over all matches (`outcome.csv`, `plots/<opponent>.png`)
    - [x] `match.matches_per_seed` (default 1): N independent matches per (learner, setting, opponent, seed), each from the pretrained weights; CIs over all matches. `--reaggregate` rebuilds outcome/chosen/plots from saved CSVs.
  - [x] Check: mean KL ≥ `kl_min` (default 0.05 nats, in `learners.yaml`) for both learners
  - [x] Tune the online LR and update frequency in `learners.yaml`. The LR must not get so high that the learner collapses: it must still beat random and must not lose more than at match start. The sweep covers lr {3e-4, 1e-3, 3e-3} × update_every {32, 64}. `chosen.csv` picks the smallest LR whose means over all matches pass KL ≥ `kl_min` and beats-random on `ppo_predicted`, and end ≥ start − 0.05 on `ppo_none` (the collapse check moved to `ppo_none` because the start score vs `ppo_predicted` is ~0, so it can't fail there).
  - [x] Log the chosen LR and its KL per seed. Full run (3 seeds × 1 match, `results/learners/main`): **lr 3e-4, update_every 64 for both learners** (already the `learners.yaml` defaults); it passes the checks on all three opponents.
    - KL vs `ppo_predicted` per seed 0/1/2: naive 1.18 / 0.36 / 0.20, reasoning 2.31 / 2.68 / 3.19 nats (vs `kl_min` 0.05)
    - Collapse check vs `ppo_none` (mean start → end score): naive 0.27 → 0.27, reasoning 0.46 → 0.44. Per seed it is not uniform: naive seed 0 drops 0.49 → 0.32.
    - End-of-match learner vs random: 0.68 (naive) and 0.66 (reasoning) on the `ppo_predicted` runs
  - [x] Test: the naive learner's win rate rises over a match vs a frozen NPC (frozen rusher, random-init learner, 150 rounds; camper/repeater are draw-heavy and give no signal)
  - [x] Test: the reasoning learner's NPC-prediction accuracy is above chance
  - [x] `scripts/train_exploiter.py`: starts from the pretrained naive learner, trains SB3 PPO for 1M steps (seeds 0–2) against the frozen `ppo_predicted` seed 0 (embedded as a `PolicyPlayer`), then evaluates it and the start policy over 200 rounds vs `ppo_predicted` and `ppo_none` (win rate, damage; greedy and stochastic). Smoke-tested.
  - [x] Full exploiter results (user run, `results/exploiter/main`; 1M steps per seed vs frozen `ppo_predicted` seed 0; player-side win rate with draws = ½, over 200 rounds; damage per round, HP 5)

    | exploiter seed | acting | vs `ppo_predicted` s0: win rate | dealt / taken | vs `ppo_none` s0: win rate | dealt / taken |
    |---|---|---|---|---|---|
    | 0 | greedy | 0.960 | 4.84 / 0.54 | 0.533 | 2.65 / 1.86 |
    | 0 | stochastic | 0.958 | 4.87 / 0.82 | 0.595 | 3.49 / 3.06 |
    | 1 | greedy | 0.448 | 2.70 / 3.41 | 0.218 | 1.69 / 3.90 |
    | 1 | stochastic | 0.513 | 2.92 / 3.33 | 0.143 | 1.69 / 4.49 |
    | 2 | greedy | 0.705 | 3.53 / 2.48 | 0.830 | 4.04 / 1.68 |
    | 2 | stochastic | 0.945 | 4.73 / 1.66 | 0.650 | 3.33 / 2.54 |
    | mean ± 95% CI | greedy | 0.704 ± 0.637 | 3.69 / 2.14 | 0.527 ± 0.761 | 2.79 / 2.48 |
    | mean ± 95% CI | stochastic | 0.805 ± 0.629 | 4.17 / 1.93 | 0.463 ± 0.692 | 2.84 / 3.36 |
    | start policy (pretrained naive) | greedy / stochastic | 0.000 / 0.000 | 0.18 / 4.99 | 0.274 / 0.262 | 2.16 / 3.79 |

    - Start-policy row: means over the 3 seeds; damage is from greedy acting.
    - `ppo_predicted` is exploitable by offline training: every exploiter seed goes from 0.00 to 0.45–0.96 against it. Seed 1 is much weaker than seeds 0 and 2, hence the wide CIs.
    - The exploit transfers only partly: against `ppo_none` (not trained on) the exploiters are level on average (0.46–0.53, start 0.26–0.27).
    - [x] Cross-evaluation vs all NPC modes and seeds: `scripts/eval_exploiters.py` → `results/exploiter/cross_eval.csv` and `cross_eval_summary.csv` (user run; 3 exploiters × 5 NPC checkpoint seeds per mode, 200 rounds per cell, stochastic acting, player-side win rate with draws = ½)
      - Pooled over the 15 cells per mode (mean ± 95% CI; damage per round dealt / taken):

        | NPC | win rate | dealt / taken |
        |---|---|---|
        | `ppo_none` | 0.43 ± 0.16 | 2.39 / 3.11 |
        | `ppo_predicted` | 0.65 ± 0.16 | 3.64 / 2.51 |
        | `ppo_oracle` | 0.11 ± 0.08 | 0.70 / 4.40 |

      - Per cell vs `ppo_predicted` (win rate; † = the cell the exploiter was trained on; damage per cell is in the CSV):

        | exploiter \ NPC seed | 0 | 1 | 2 | 3 | 4 | mean of unseen (1–4) |
        |---|---|---|---|---|---|---|
        | exploiter_seed0 | 0.920 † | 0.338 | 0.995 | 0.935 | 0.193 | 0.615 |
        | exploiter_seed1 | 0.578 † | 0.740 | 0.933 | 0.740 | 0.690 | 0.776 |
        | exploiter_seed2 | 0.950 † | 0.413 | 0.073 | 0.795 | 0.533 | 0.453 |

        The † cells differ slightly from the `results/exploiter/main` table above (e.g. 0.920 vs 0.958) because the cross-eval uses different env seeds.
      - Finding: the exploit transfers on average (about 0.61 over the 12 unseen `ppo_predicted` cells, 8 of 12 above 0.5) but unevenly (0.07 to 0.995), and different exploiters beat different NPC seeds. For example, NPC seed 2 loses 0.995 and 0.933 to exploiters 0 and 1 but beats exploiter 2 (0.073), while NPC seed 4 beats exploiter 0 (0.193) but not exploiter 1 (0.690). Against `ppo_oracle`, which sees the exploiter's true next action, the exploit fails (0.11).
  - [ ] (Optional) Retrain the NPC with learner snapshots in the opponent pool
  - [x] Commit

### Tier 3 — Experiments
- [ ] **M10 Experiment grid**
  - [ ] Methods: scripted, PPO, pred-PPO, MBOM, and 4 ablations
  - [ ] Opponents: fixed archetypes, switcher, naive, reasoning, held-out, frozen exploiter
    - Frozen exploiter: `models/exploiter/exploiter_seed{0,1,2}.pt` (from `scripts/train_exploiter.py`), loaded as a `PolicyPlayer` with stochastic acting, no online updates. It is a known counter to `ppo_predicted`, so it tests whether an NPC (in particular MBOM) can adapt to an opponent that already exploits it.
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
   - Train: rusher, camper, pattern repeater, random mixer, flanker (reaches a side tile of the opponent and attacks from there), switcher (cycles through rusher, camper, repeater and mixer, as set in `players.yaml`).
   - Held-out: kiter (hit and retreat), counter-puncher (blocks until attacked, then punishes; `reaction_delay: 1` and `noise: 0.2` after M3 balancing).
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

## Known limitations
- **Flanker is weak overall.** It beats `counter_puncher` (about 64%), repeater and mixer, but loses to nearly everything else (for example about 1% against rusher and 2% against kiter).
- **Passive archetypes mostly draw.** Camper, repeater, mixer and random draw against each other most of the time, so 20 pairs are above 50% draws. The draw-adjusted win rates for those pairs sit close to 0.5.
- **`counter_puncher` still wins 90%+ against deterministic opponents** (rusher, kiter, repeater, mixer). It is no longer dominant, because it loses to flanker and `scripted_npc`. Softening only the delay and noise cannot fix this. Getting closer to 60–70% would need a change in its logic.
- **Win rate saturates at about 100% against the fixed archetypes** for all three PPO modes (none, predicted, oracle), so win rate cannot separate them there. The comparison relies on the other metrics (episode length, damage taken, win rate over the first N rounds, prediction accuracy) and on the learning and switching opponents.
- **Learners can't beat `ppo_predicted`.** Over a 30-round match, `ppo_predicted` wins about 90–100% against every learner and every LR / update-frequency setting (learner scores 0.00–0.10; verdict "lose" for all). Online learning shows up only as small within-match gains (last 10 − first 10 rounds up to about +0.17). Against `ppo_none` the learners are level (verdict "match", scores 0.2–0.6 with wide CIs); against `scripted` they win 0.88–1.00.
- **The exploit exists but is out of reach online.** Offline exploiters, starting from the same pretrained naive policy, beat `ppo_predicted` (win rate 0.70–0.81 mean over 3 seeds), but they needed 1M training steps. A 30-round match gives an online learner only about 600–1,000 steps, roughly 0.1% of that budget, so the learners can't find the exploit within a match. The online learners' "lose" verdict vs `ppo_predicted` reflects the match budget, not an unexploitable NPC.
- **Cross-eval cells aren't independent samples.** The 15 cells per NPC mode cross 3 exploiters with 5 NPC checkpoints, so every exploiter appears in 5 cells and every NPC seed in 3. The pooled 95% CI treats the 15 cells as independent (t-interval, 14 df), so it understates the uncertainty. The real sample sizes are 3 exploiters and 5 NPC seeds, and the cell-to-cell spread (0.07–0.995 vs `ppo_predicted`) mixes exploiter effects, NPC-seed effects and their interaction, which 15 cells can't separate. Read the pooled numbers as descriptive, and use the per-cell table.
- **KL is uninformative for near-deterministic policies.** The pretrained learners' entropy is only about 0.07–0.3 nats (maximum ln 8 ≈ 2.08). With so little probability mass off the argmax, small logit changes give large KL(π_start ‖ π_end): even the smallest LR gives 0.2–3.2 nats, far above `kl_min` = 0.05, so the threshold passes trivially. KL is reported, but the per-round score, damage and entropy are the meaningful signals. KL above `kl_min` does not mean the learner adapted usefully.

---

## Verification (every milestone)
- `pytest -q` passes, with new tests added each milestone.
- `python -m scripts.benchmark_env --config configs/env.yaml` reports steps/s for the single and vectorized envs.
- `python -m scripts.balance_matrix --config configs/balance.yaml` writes the matrix and the pass/fail result of the dominance check.
- `python -m scripts.check_learner_drift --config configs/learners.yaml` reports the start→end KL per learner, plus the beat/match/lose outcome per opponent.
- Each training script has a smoke config (`configs/experiments/smoke.yaml`). I run it end to end and check that `results/.../metrics.csv` is written. The user runs the full-budget runs and pastes summaries.
- `python -m scripts.play --tick-rate 8` for manual checks of rendering, controls and the demo panel.
- Commit after each green milestone, for example `M3: game balance matrix + tuned env params`.

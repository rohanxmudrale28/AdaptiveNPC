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
- [x] **M7 Env model Γ**
  - [x] `models/env_model.py` (Γ, `TrueSimulator`, `rollout`), `eval/env_model_eval.py`, `scripts/collect_env_data.py`, `scripts/train_env_model.py`, `configs/env_model.yaml` (+ smoke config)
  - [x] State = 15 integers (positions, facing, hp, stamina, cooldown, stun, t). Every discrete channel is a classification head (delta class per channel, 4-way facing, t = t+1), snapped to a legal integer state that feeds back in rollouts. Reward: Γ learns the shaping part; the win/loss/draw bonus is computed from the predicted next state (a rare ±1 spike was unlearnable as MSE: first regression version had 0% terminal-reward accuracy and 75% full-state exactness).
  - [x] Data: NPCs scripted, random, ppo_none/predicted/oracle (5 checkpoint seeds, stochastic) × opponents (train archetypes, random, online naive/reasoning learners, exploiters 0–1). `train` / `val` use the same opponents with different env seeds; `heldout_opp` = kiter, counter_puncher, exploiter_seed2. 86.7k / 34.3k / 23.2k transitions.
  - [x] Random-action augmentation (`models/env_model_data.py`, `aug:` block in `env_model.yaml`): start states sampled only from the train rollouts (seed 11, never the val states or eval seed 7), both sides uniformly random, exact labels from the true simulator, half of it as chained 2-step rollouts, collision-heavy starts (near / on the edge) sampled 2x. 57.7k random-action + 86.7k on-policy = 40% of the training mix.
  - [x] Full run (`results/env_model/main`, 40 epochs). Full-state exact, 1-step / 2-step: val 99.38% / 98.79%; held-out opponents 99.29% / 98.63%; random-action slice (20k val states, both sides random) 98.89% / 97.27% (before augmentation: 98.68 / 97.45, 97.53 / 95.45 and 93.73 / 88.98). Reward within 0.05: 99.3% / 99.0% / 99.0% (1-step). Tables: `error_table.md`, `errors.csv`, `errors_by_group.csv`; round-end reward error by outcome: `terminal_reward.csv`.
  - Round-end reward (1-step, within 0.05): KO wins 98.5%, timeouts and draws 100%, KO losses 85.9% (val; 87.2% held-out opponents; 50% of 16 on the random slice). Γ gets the outcome and the end of the round right (99.2% / 100%); the residual error is the ±0.1 shaping term on the step where the NPC dies.
  - [x] Tests (`tests/test_env_model.py`): oracle simulator replays the core exactly; Γ rollouts vs the true simulator; checkpoint round trip. `oracle_model` flag: `--set oracle_model=true` evaluates the true simulator (100% by construction).
  - [x] Commit
- [ ] **M8 MBOM**
  - [x] `npc/mbom/state_obs.py`: batched raw state → observation for imagined states (bit-exact vs `encode`)
  - [ ] Distribution-shift probe, `scripts/probe_cond_shift.py` (config `probe:` in `configs/mbom.yaml`). The frozen `ppo_predicted` gets a transformed conditioning vector instead of raw φ0, vs the train archetypes, 5 NPC seeds × 30 rounds, same env seeds per variant (paired). Variants: φ0 (control), smoothed toward uniform (λ 0.25 / 0.5), `mix_sharp` (½ φ0 + ½ sharpened φ0), `mix_perm` (½ φ0 + ½ φ0 with permuted actions), `mix_alpha_cap` (0.79 φ0 + 0.21 permuted), oracle one-hot.
    - **Pass criterion (fixed before the run):** switch `mbom.cond_input` from `mix` to `top_level` or `sample_level` if the pooled win rate (draws = ½) vs the train archetypes of any mixed-input variant (`mix_sharp`, `mix_perm`, `mix_alpha_cap`) drops more than 5 points below raw φ0 (paired mean drop > 0.05). Otherwise keep `mix`.
    - Caveat: `ppo_predicted` wins about 100% vs the train archetypes, so win rate can only fall. Damage taken and the action-change rate are reported as well, but they don't enter the decision.
    - **Result (user run, `results/mbom/probe_cond_shift`; 30 paired cells per variant = 6 train archetypes × 5 NPC seeds, 30 rounds each): KEEP `cond_input: mix`.** The worst variant, `mix_perm`, drops 1.4 points (< 5). A secondary check, set by the user, did not trigger either: action change above 25%, or damage taken up more than 0.5 HP over φ0. So the M10 grid gets no extra `top_level` variant.

      | variant | win rate | drop vs φ0 ± 95% CI | damage taken / round | action change |
      |---|---|---|---|---|
      | φ0 (control) | 1.000 | – | 0.16 | 0.000 |
      | smooth25 | 1.000 | 0.000 ± 0.000 | 0.15 | 0.049 |
      | smooth50 | 0.999 | 0.001 ± 0.002 | 0.18 | 0.105 |
      | mix_sharp † | 0.999 | 0.001 ± 0.002 | 0.17 | 0.034 |
      | mix_perm † | 0.986 | 0.014 ± 0.015 | 0.44 | 0.191 |
      | mix_alpha_cap † | 0.999 | 0.001 ± 0.002 | 0.16 | 0.068 |
      | oracle | 1.000 | 0.000 ± 0.000 | 0.12 | 0.114 |

      † = decision variant.
    - Win rate is saturated here, so damage taken and action change are what carry the information. Confident disagreement between levels (`mix_perm`) changes about 1 in 5 actions and nearly triples damage taken (0.16 → 0.44 HP per round), yet the NPC still wins. A mixture that agrees with φ0 (`mix_sharp`, `mix_alpha_cap`) barely matters. The policy is robust enough for `mix` against these opponents. Against stronger opponents (exploiters, learners) the same shift may cost more, and the per-round drift log tracks this.
  - [x] `iops.py`: φ0 is the pretrained predictor, finetuned online (lr 1e-3, 3 steps every 50 real steps on the last 256). φ1 and φ2 start each match as copies of φ0 and are finetuned incrementally on soft best-response targets softmax(Q/τ), τ = 0.1. The first update uses 10 steps (`warmup_grad_steps`).
  - [x] `imagination.py`: eq. (5), batched in torch (`EnvModel.step_t` keeps Γ in torch; `step` is now a wrapper around it, bit-identical on 200k rows)
    - `score_agg: max | mean`. The opponent continuations and the NPC's sampling noise are drawn once per state and shared by all 8 candidate actions (common random numbers). `max` (default) takes the mean over NPC samples, then the max over opponent continuations.
    - The sharing costs nothing, but each extra NPC sample adds a full rollout pass: one level on 256 states takes 1.5 s at `npc_samples: 1` and 5.9 s at 4 (10 threads). So the default is `npc_samples: 1`, where the NPC's noise is shared across candidates but not averaged.
    - `imagined_npc: stochastic | greedy` sets the NPC action convention in imagination. It is recorded in the results config together with the evaluation convention (`conventions:` block written by `eval_mbom.py`).
  - [x] `mixer.py`: eq. (6–8), with an ε-floor on p(aᵒ) and a prior floor (0.01) so a level that has lost for a long time can recover.
    - Measured on synthetic data (20 seeds): the generating level leads for good by step 17.
    - After a switch the new level leads for good after a median 50 steps (max 98) when levels overlap strongly (peak 0.4), and after 20 (max 26) at peak 0.6. The lag comes from the sticky prior (moving average of posteriors), not from the evidence. `prior_avg` and `prior_floor` are tuning knobs (dev opponents only).
    - Steady-state α is 0.71–0.77, under the 0.787 cap.
  - [x] `agent.py`: online loop (Alg. 1). The policy is the frozen `predicted` PPO, fed the mixed IOP. `MBOMEnv` wraps `NPCEnv` like `ConditionedEnv`. The runner methods are `mbom`, `mbom_<ablation>` and `mbom_single_level<m>`, and `scripts/eval_mbom.py` is the CLI.
  - [x] Ablation flags: `no_imagination`, `random_finetune` (BM), `single_level=m`, `uniform_mix`
  - [x] Per-round log (`mbom_rounds.csv` per cell; step-weighted match means in `metrics.csv`):
    - mixer weights α over the levels (mean and end of round), with an `alpha_cap` column (≈ 0.79 for T = 0.5)
    - per-level and mixed IOP prediction accuracy, and mixed NLL
    - policy input drift: entropy and max-prob of the policy input vs φ0, and the fraction of steps where the policy's argmax differs from π(·|φ0)
    - updates, update seconds, Q spread
  - [x] Test: the mixer converges to the generating level on synthetic data, follows a switch, respects the α cap, and the mode flags hold α fixed
  - [x] Test: imagination finds the known best response in two hand-built states. Ground truth is brute force over all continuations with the true rules, and the learned Γ gives the same argmax and Q. Also tested: the done mask, the shared NPC samples, determinism and chunk independence.
  - [x] Tests: ablation flags, `cond_input` modes, agent determinism, a grid smoke run (deterministic across two runs)
  - [x] `scripts/benchmark_mbom.py`: seconds per update (Γ / GRU / policy breakdown) and per 30-round match, plus an M10 projection (`results/benchmarks/mbom_speed.csv`, `mbom_m10_projection.csv`)
    - Measured (user run; NPC seed 0, 30 rounds vs rusher / switcher / kiter / counter_puncher; 557–1,126 real steps per match):

      | method | per match, 10 threads | per match, 1 thread | per update, 10 / 1 threads |
      |---|---|---|---|
      | `mbom` | 23–56 s (mean 38) | 48–113 s (mean 77) | ≈ 2.0–2.7 s / 4.2–5.5 s |
      | `mbom_single_level1` | 16–30 s (mean 21) | 29–59 s (mean 43) | ≈ 1.25 s / 2.6 s |
      | `mbom_random_finetune` | 3–6 s | 3–6 s | ≈ 0.12 s |
      | `mbom_no_imagination` | 1–4 s | 2–4 s | ≈ 0.02–0.04 s |
      | `ppo_predicted` | 1.5–2.7 s | 1.4–2.6 s | – |

    - A full update on a 256-entry buffer takes ≈ 3.0 s at 10 threads and ≈ 6.3 s at 1 thread, against the analytic estimate of 1–3 s. One thread is only about 2× slower than 10, so parallel processes at 1 thread each are the efficient way to run the grid.
    - **The IOP GRU dominates update time:** 1.8 s of 3.0 (≈ 60%; 4.0 of 6.3 s at 1 thread). Γ takes 0.7–0.9 s (≈ 27%), the policy ≈ 0.1 s, and the rest (finetuning, bookkeeping) 0.24–0.45 s. The obvious speedup, if needed, is the 8-step GRU window recomputed on every imagined step.
    - **`random_finetune` is cheap because it skips imagination.** It only finetunes on random labels, about 0.12 s per update, so it costs about as much as `no_imagination` and doesn't need the same compute as MBOM.
    - `single_level=1` costs about half of MBOM (one imagined level instead of two). `single_level=2` and `uniform_mix` build both levels and cost the same as MBOM.
  - [ ] Go/no-go, `scripts/gonogo_mbom.py`: MBOM vs `ppo_predicted` vs `no_imagination` on the development exploiter (`exploiter_seed1`) against NPC seeds 0–4 (user run)
    - **Criterion (fixed before any run):**
      - Design: 5 cells (NPC seeds 0–4) × 7 matches per cell. A match is 30 rounds with fresh MBOM state (IOPs, mixer, buffer); the `ppo_predicted` predictor is also fresh, and the exploiter is frozen. That gives 210 rounds per cell per method.
      - Methods: `mbom` (default config), `mbom_no_imagination`, and `ppo_predicted` with its own online predictor setting (`predictor.yaml` `online`).
      - Pairing: match i of a cell uses the same env seed and exploiter seed for every method, so methods are compared match by match.
      - Win rate counts draws as ½.
      - **GO** if MBOM's mean win rate over the 5 cells beats `ppo_predicted`'s by at least 5 points, **and** the lower bound of the 95% t-interval of the paired per-match difference (MBOM − `ppo_predicted`, 35 matches) is above 0. Otherwise **NO-GO**.
    - Also reported, but not part of the decision:
      - MBOM vs `no_imagination`, with the same paired statistics. This isolates imagination, since both share the φ0 online schedule.
      - Each method's win rate in rounds 1–5 vs rounds 26–30 (adaptation within a match).
      - Per-cell win rates, damage, and the MBOM trace means (α, per-level accuracy, input drift).
    - Caveats:
      - **Development signal, not a final claim.** `exploiter_seed1` is the development exploiter, so the go/no-go result and any tuning it triggers are excluded from the final claims (which use `exploiter_seed0` / `exploiter_seed2`).
      - The 7 matches in a cell share one NPC checkpoint, so the 35 paired differences aren't independent and the CI understates the uncertainty (as in the cross-eval). Per-cell differences are reported too.
    - **Result (user run, `results/mbom/gonogo`): NO-GO.** MBOM beats `ppo_predicted` by 0.004 (paired 0.004 ± 0.034, CI low −0.031; needs ≥ 0.05 and CI low > 0). **MBOM trails `no_imagination` by 0.050 ± 0.031** (paired over 35 matches, CI low −0.081). Imagination made things worse here. Win rates are NPC-side, draws = ½.

      | method | win rate | ± 95% CI (over cells) | rounds 1–5 | rounds 26–30 | damage taken |
      |---|---|---|---|---|---|
      | `mbom` | 0.454 | 0.303 | 0.286 | 0.554 | 3.22 |
      | `mbom_no_imagination` | 0.504 | 0.288 | 0.309 | 0.566 | 2.98 |
      | `ppo_predicted` | 0.450 | 0.299 | 0.337 | 0.526 | 3.24 |

      | comparison | mean gain over cells | paired mean ± 95% CI (35 matches) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
      |---|---|---|---|---|---|---|---|
      | mbom − ppo_predicted | +0.004 | +0.004 ± 0.034 | +0.002 | +0.043 | −0.038 | −0.007 | +0.019 |
      | mbom − mbom_no_imagination | −0.050 | −0.050 ± 0.031 | −0.021 | +0.010 | −0.167 | −0.043 | −0.029 |

      Per cell (win rate mbom / no_imagination / ppo_predicted): seed 0 0.410 / 0.431 / 0.407; seed 1 0.243 / 0.233 / 0.200; seed 2 0.231 / 0.398 / 0.269; seed 3 0.579 / 0.621 / 0.586; seed 4 0.810 / 0.838 / 0.790. Most of the MBOM deficit comes from seed 2 (−0.167 vs `no_imagination`, where rounds are longest, 87 steps). All three methods improve from rounds 1–5 to 26–30, by +0.19 to +0.27.
    - **Diagnosis (from the saved go/no-go data, no new runs; `scripts/diagnose_mbom.py` → `results/mbom/gonogo/diagnosis/`)**, step-weighted, rounds 1–5 / 6–25 / 26–30:

      | | rounds 1–5 | rounds 6–25 | rounds 26–30 |
      |---|---|---|---|
      | φ0 accuracy (mbom) | 0.339 | 0.647 | 0.783 |
      | φ1 / φ2 accuracy (mbom) | 0.168 / 0.160 | 0.167 / 0.178 | 0.178 / 0.206 |
      | φ1 = φ2 = untouched pretrained copies (no_imagination) | 0.256 | 0.269 | 0.270 |
      | α on φ0 / φ1 / φ2 | 0.52 / 0.24 / 0.24 | 0.76 / 0.12 / 0.12 | 0.78 / 0.11 / 0.11 |
      | policy input entropy: mixed vs φ0 | 1.88 vs 1.72 | 1.57 vs 1.23 | 1.33 vs 0.85 |
      | policy input max-prob: mixed vs φ0 | 0.29 vs 0.36 | 0.47 vs 0.57 | 0.59 vs 0.72 |
      | steps where the policy's argmax changes vs π(·\|φ0) | 3.6% | 5.7% | 6.1% |
      | Q spread (max − min over the 8 candidates) per update | 0.149 | 0.205 | 0.188 |
      | implied max soft-BR target prob, e^{s/τ}/(e^{s/τ}+7) | 0.39 | 0.53 | 0.48 |

      1. **The best-response levels predict this frozen exploiter worse than φ0, and worse than doing nothing.** φ0 learns the exploiter online (0.34 → 0.78). φ1 and φ2 stay at 0.16–0.21, below the untouched pretrained copy (0.26–0.27). The k = 2 imagined best response against π∘φ_{m−1} in Γ doesn't resemble what this exploiter does. That is expected for a fixed policy trained offline: it isn't reasoning about the current NPC model.
      2. **The mixer does its job but can't fully commit.** α moves to φ0 and saturates at the 0.787 cap from round ~6. The remaining ≈ 22% on φ1 and φ2 flattens the policy input more and more as φ0 sharpens (entropy gap 0.16 → 0.48 nats; max-prob 0.72 → 0.59), and the policy's argmax flips on about 6% of late steps. The effect is largest on seed 2, where φ0 is most accurate: 0.86 accuracy, entropy 0.60 vs 1.18 mixed, 7.5% flips. That is the cell with the −0.167 deficit. So MBOM behaves like a blurred `no_imagination`.
      3. **The Q spread isn't zero (0.15–0.21), so the BR targets aren't pure noise.** At τ = 0.1 they are soft, though: the best action gets at most 0.39–0.53. Combined with (1), the targets are informative about the imagined opponent but wrong about the real one.
      4. **Early rounds:** the first update comes in round 2 (median; range 1–3), at real step 50. Before it, MBOM and `no_imagination` played identical rounds in 35/35 matches (all levels equal φ0, and the pairing and determinism hold). After it, MBOM falls behind: rounds 1–5 win rate 0.286 vs 0.309. `ppo_predicted` is best early (0.337), probably because its own online setting updates sooner (every 25 steps), but it is the worst late.
      - Prediction for the variants: `top_level` should recover `no_imagination` (it feeds φ0 alone when α favours it). The prior settings shouldn't matter much, because the cap comes from the temperature T, not the prior. Lowering T (a higher cap) is the direct fix for (2). It is proposed, not run.
    - **Adaptive-opponent hypothesis test (pre-registered before any run; `scripts/dev_variants.py --adaptive`, `results/mbom/adaptive_test`):**
      - Question: does imagination help against opponents that adapt? Opponents: `naive_learner`, `reasoning_learner` (online, chosen lr3e-4_u64), `switcher`. 5 NPC seeds × 7 matches × 30 rounds, fresh state per match, paired seeds across methods.
      - **H1:** mbom − `no_imagination` > 0 on pooled paired win rate (95% CI low > 0). Also reported per opponent and vs `ppo_predicted`.
      - Secondary: paired damage taken (lower is better), because `ppo_predicted` wins about 100% against learners (M9) and the switcher (M6), so win rate is likely saturated. If all methods are at ≥ 0.95, H1 is reported as untestable on win rate and the conclusion rests on damage taken.
      - This is not a decision gate. Learners and the switcher aren't exploiters 0 or 2, so this stays within the development set.
    - **Dev-variant results (user run; `results/mbom/dev_variants.csv`, `scripts/dev_variants.py`; development set only).** Every variant tried is listed; **none failed** (status ok for all 10 variants × 2 designs).
      - Exploiter design: `exploiter_seed1`, 5 NPC seeds × 7 matches, paired with the go/no-go seeds. The three references are the reused go/no-go runs.
      - Diff columns: paired win-rate difference ± 95% CI over 35 matches. Win rate is NPC-side, draws = ½. CPU = single-thread minutes, both designs.

      | variant | overrides | win rate | rounds 1–5 | 26–30 | damage | mixed acc | argmax changed | vs mbom | vs no_imagination | vs ppo_predicted | CPU min |
      |---|---|---|---|---|---|---|---|---|---|---|---|
      | mbom (ref) | – | 0.454 | 0.286 | 0.554 | 3.22 | 0.608 | 0.053 | – | −0.050 ± 0.031 | +0.004 ± 0.034 | reused |
      | mbom_no_imagination (ref) | – | 0.504 | 0.309 | 0.566 | 2.98 | 0.589 | 0.000 | +0.050 ± 0.031 | – | +0.054 ± 0.039 | reused |
      | ppo_predicted (ref) | – | 0.450 | 0.337 | 0.526 | 3.24 | – | – | −0.004 ± 0.034 | −0.054 ± 0.039 | – | reused |
      | top_level | `cond_input=top_level` | 0.501 | 0.311 | 0.577 | 3.00 | 0.595 | 0.001 | +0.047 ± 0.032 | −0.003 ± 0.012 | +0.051 ± 0.039 | 336 + 126 |
      | npc_samples4 | `npc_samples=4` | 0.442 | 0.286 | 0.531 | 3.22 | 0.607 | 0.053 | −0.012 ± 0.024 | −0.062 ± 0.035 | −0.008 ± 0.036 | 1145 + 312 |
      | uniform_mix | `mbom_uniform_mix` | 0.336 | 0.263 | 0.400 | 3.72 | 0.588 | 0.149 | −0.119 ± 0.054 | −0.169 ± 0.061 | −0.115 ± 0.055 | 324 + 140 |
      | single_level1 | `mbom_single_level1` | 0.252 | 0.289 | 0.223 | 4.02 | 0.181 | 0.211 | −0.202 ± 0.050 | −0.252 ± 0.059 | −0.199 ± 0.056 | 162 + 94 |
      | prior_fast | `prior_avg=0.7, prior_floor=0.05` | 0.453 | 0.297 | 0.549 | 3.21 | 0.606 | 0.056 | −0.001 ± 0.027 | −0.051 ± 0.036 | +0.002 ± 0.034 | 335 + 127 |
      | prior_slow | `prior_avg=0.97, prior_floor=0.002` | 0.431 | 0.263 | 0.523 | 3.32 | 0.613 | 0.055 | −0.023 ± 0.028 | −0.073 ± 0.035 | −0.020 ± 0.034 | 357 + 116 |
      | oracle_debug (**debug, not reported**) | `oracle_model=true` | 0.458 | 0.286 | 0.571 | 3.18 | 0.608 | 0.052 | +0.003 ± 0.021 | −0.047 ± 0.031 | +0.007 ± 0.038 | 368 + 113 |

      - Archetype design (6 train archetypes × 5 NPC seeds × 1 match): win rate is saturated (0.996–1.000 for every variant, all paired differences within ±0.005).
        - Damage taken per round: mbom 0.16, no_imagination 0.16, ppo_predicted 0.14, top_level 0.15, npc_samples4 0.15, prior_fast 0.16, prior_slow 0.16, uniform_mix 0.21, single_level1 0.44.
        - Argmax changed: mbom 0.045, top_level 0.0001, uniform_mix 0.14, single_level1 0.23.
      - **Pre-registered predictions:**
        - **Held:** `top_level` recovers `no_imagination` (−0.003 ± 0.012). With φ0 fed alone, argmax flips drop from 5.3% to 0.1%.
        - **Held:** the prior settings don't matter much. `prior_fast` is −0.001 ± 0.027 vs mbom; `prior_slow` is −0.023 ± 0.028, a slight drop, CI includes 0.
        - Lowering T was proposed only, not run.
      - **Further findings:**
        - Γ error is not the cause. Imagination with the true simulator (`oracle_debug`) matches mbom (+0.003 ± 0.021) and still trails `no_imagination` (−0.047 ± 0.031). The deficit comes from the best-response concept against this opponent, not from model error.
        - Averaging the NPC's sampling noise doesn't help (`npc_samples4`: −0.012 ± 0.024 vs mbom at 3.5× the CPU).
        - The mixer has value relative to not mixing. Uniform weights lose 0.119 and committing to φ1 alone loses 0.202 vs mbom, because the BR levels are bad predictors of this opponent (mixed accuracy 0.18 with φ1 alone). The mixer's job here is to down-weight them, and the α cap stops it from finishing the job.
    - **Adaptive-opponent test result (H1; user run, `results/mbom/adaptive_test/summary.csv`): H1 not supported.**
      - Win rate is not saturated against the learners (0.87–0.93; it falls over the match as the learners adapt, e.g. naive 0.94 → 0.77), so H1 is testable on win rate. It is saturated against the switcher (≈ 1.00).
      - Pooled over 105 paired matches: mbom − `no_imagination` = −0.011 ± 0.020 (CI low −0.031); damage taken −0.011 ± 0.097 (no difference).

      | opponent | mbom (1–5 → 26–30) | no_imagination | ppo_predicted | mbom − no_imag | mbom − ppo_pred | damage: mbom / Δ vs no_imag |
      |---|---|---|---|---|---|---|
      | naive_learner | 0.866 (0.94 → 0.77) | 0.890 (0.96 → 0.77) | 0.885 (0.92 → 0.79) | −0.024 ± 0.041 | −0.019 ± 0.042 | 1.11 / +0.11 ± 0.19 |
      | reasoning_learner | 0.899 (0.94 → 0.85) | 0.905 (0.95 → 0.82) | 0.910 (0.95 → 0.83) | −0.007 ± 0.048 | −0.011 ± 0.042 | 1.14 / −0.15 ± 0.22 |
      | switcher | 0.998 (1.00 → 1.00) | 0.999 | 0.999 | −0.001 ± 0.002 | −0.001 ± 0.002 | 0.18 / +0.01 ± 0.03 |
      | pooled (105 matches) | 0.921 (0.96 → 0.87) | 0.932 | 0.931 | −0.011 ± 0.020 | −0.010 ± 0.019 | 0.81 / −0.01 ± 0.10 |

      Imagination doesn't help against these adapting opponents. The learners adapt by online PPO (slowly, M9) rather than by reasoning about the NPC, so a k = 2 best response isn't what they converge to either. No defaults were changed.
    - **Learner diagnosis (from the saved adaptive-test traces, no new runs; `scripts/diagnose_mbom.py --test adaptive_test` → `results/mbom/adaptive_test/diagnosis/phases.csv`)**, step-weighted, rounds 1–5 / 6–25 / 26–30:

      | | naive_learner | reasoning_learner |
      |---|---|---|
      | φ0 accuracy (mbom) | 0.396 / 0.727 / 0.836 | 0.387 / 0.646 / 0.750 |
      | φ1 accuracy (mbom) | 0.125 / 0.123 / 0.148 | 0.098 / 0.144 / 0.159 |
      | φ2 accuracy (mbom) | 0.139 / 0.157 / 0.194 | 0.125 / 0.159 / 0.168 |
      | untouched pretrained copy (no_imagination φ1 = φ2) | 0.282 / 0.362 / 0.410 | 0.257 / 0.291 / 0.301 |
      | α on φ0 / φ1 / φ2 | 0.59/0.20/0.21 → 0.78/0.11/0.11 → 0.78/0.11/0.11 | 0.65/0.17/0.18 → 0.77/0.11/0.11 → 0.78/0.11/0.11 |
      | policy input entropy, mixed vs φ0 | 1.78 vs 1.53 / 1.36 vs 0.86 / 1.18 vs 0.57 | 1.73 vs 1.45 / 1.42 vs 0.94 / 1.29 vs 0.73 |
      | policy input max-prob, mixed vs φ0 | 0.35 vs 0.43 / 0.57 vs 0.69 / 0.66 vs 0.80 | 0.36 vs 0.45 / 0.53 vs 0.65 / 0.60 vs 0.74 |
      | steps where the policy's argmax changes | 3.1% / 4.6% / 5.1% | 3.7% / 6.9% / 7.4% |
      | Q spread per update (implied max BR target prob) | 0.078 / 0.089 / 0.096 (0.24–0.27) | 0.070 / 0.078 / 0.088 (0.22–0.26) |

      - **Yes, φ0 still beats φ1 and φ2 against both learners, by a wide margin.** φ0 reaches 0.84 / 0.75 late, while φ1 and φ2 stay at 0.10–0.19, around chance (1/8 = 0.125) and well below the untouched pretrained copy (0.30–0.41). This is the same pattern as against the exploiter.
      - The mixer again moves to φ0 and sits at the 0.78 cap from round ~6. The remaining 22% widens the gap between the mixed input and φ0's (entropy gap 0.25 → 0.61 nats vs naive), and the policy's argmax flips on 5–7% of late steps.
      - The best-response targets are flatter than against the exploiter. The Q spread is 0.07–0.10 (vs 0.15–0.21), so the best action gets at most 0.22–0.27 vs uniform 0.125. Imagination barely distinguishes the candidate actions here.
      - Side note, switcher (same CSV): the untouched pretrained copy predicts the switcher better than the online-finetuned φ0 late in the match (0.96 vs 0.91), so the φ0 finetune (lr 1e-3) slightly hurts against known archetypes.
    - **NPC win rate per round vs the learners** (`scripts/plot_adaptive_rounds.py` → `results/mbom/adaptive_test/win_rate_by_round.png` / `.csv`; 35 matches per point, per-point 95% CI ≈ ±0.08):

      | | rounds 1–5 | rounds 11–20 | rounds 26–30 |
      |---|---|---|---|
      | naive: mbom / no_imagination / ppo_predicted | 0.937 / 0.963 / 0.920 | 0.876 / 0.901 / 0.910 | 0.769 / 0.769 / 0.786 |
      | reasoning: mbom / no_imagination / ppo_predicted | 0.943 / 0.946 / 0.954 | 0.900 / 0.926 / 0.931 | 0.846 / 0.823 / 0.829 |

      All three NPCs decline steadily over the match as the learners adapt, and the curves overlap within the per-round noise. The decline hasn't levelled off by round 30.
    - **Lookahead opponent and H2 (pre-registered before any run; `adaptive_npc/players/lookahead.py`, `lookahead:` / `h2_test:` in `configs/mbom.yaml`).** This is the test imagination should pass if it works.
      - The lookahead opponent assumes the NPC is plain `ppo_predicted`. Each step it predicts the NPC's action with the cell's frozen `ppo_predicted` checkpoint, fed by the pretrained, frozen φ0.
      - It then picks the action that maximizes the eq.-5 imagined return from its own side. It reuses `imagined_returns` with MBOM's settings: k = 2 random continuations, 16 samples, stochastic NPC, max over continuations, bootstrap −γ³V. It runs in the true simulator, reconstructs the raw state exactly from its observation, and plays a uniformly random action with ε = 0.05. No online learning.
      - In other words, it is a level-1 reasoner against π∘φ0, which is exactly what φ1 is built to model.
      - **H2:** against the lookahead opponent, MBOM beats `no_imagination` by at least 0.05 (mean win rate over the 5 cells), **and** the 95% t-interval of the paired per-match difference is above 0. Design: NPC seeds 0–4, 7 matches × 30 rounds per cell, fresh state per match, paired seeds (env seed base 50000), with `ppo_predicted` also reported.
      - The opponent's settings are fixed now and **will not be tuned after the run**, whatever the result.
      - `scripts/dev_variants.py --test h2_test` prints SUPPORTED / NOT SUPPORTED.
    - **H2 result (user run, `results/mbom/h2_lookahead/summary.csv`): NOT SUPPORTED.** mbom − `no_imagination` = −0.007 over cells; paired −0.007 ± 0.041 over 35 matches, CI low −0.048 (needed ≥ +0.05 and CI low > 0). The opponent and the criterion are unchanged.

      | method | win rate | rounds 1–5 → 26–30 | damage taken (of 5 HP) |
      |---|---|---|---|
      | `mbom` | 0.116 | 0.106 → 0.126 | 4.67 |
      | `mbom_no_imagination` | 0.122 | 0.111 → 0.094 | 4.65 |
      | `ppo_predicted` | 0.098 | 0.077 → 0.097 | 4.73 |

      | comparison (35 paired matches) | win rate | damage taken |
      |---|---|---|
      | mbom − `no_imagination` | −0.007 ± 0.041 | +0.02 ± 0.09 |
      | mbom − `ppo_predicted` | +0.018 ± 0.032 | −0.06 ± 0.07 |

      - **Floor effect:** every method won only about 0.10–0.12 of rounds against the lookahead opponent and lost about 4.7 of 5 HP per round. The lookahead opponent dominates all three NPCs, so there is little room for any of them to differ. H2 therefore says imagination doesn't rescue the NPC against an opponent this much stronger. It can't say whether imagination helps against a level-1 reasoner the NPC can compete with.
      - **H1 and H2 were both not supported.** Imagination gave no detectable benefit against adaptive learners (H1) or against the lookahead opponent it is designed to model (H2). Per Amendment 1, `mbom_single_level2` and `mbom_random_finetune` are **not** run on the final grid.
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
    - The exploit transfers only partly: against `ppo_none` (not trained on) the exploiters are level on average (0.46–0.53, start 0.26–0.27). **Corrected in M10:** on NPC seeds 1–4 the exploiters beat `ppo_none` about as much as `ppo_predicted` (difference +0.046 ± 0.130). Only the seed-0 `ppo_predicted` checkpoint is specifically exploited.
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
- [x] **M10 Experiment grid**
  - [ ] Methods:
    - baselines: `scripted`, `ppo_none`, `ppo_predicted` (with its own online predictor setting)
    - MBOM and its ablations: `no_imagination`, plus five imagination methods (`mbom`, `random_finetune`, `single_level=1`, `single_level=2`, `uniform_mix`)
    - `ppo_predicted` vs `no_imagination` isolates the φ0 online schedule, and `no_imagination` vs `mbom` isolates what imagination adds.
  - [ ] Opponents: fixed archetypes, switcher, naive, reasoning, held-out, frozen exploiter
    - Frozen exploiter: `models/exploiter/exploiter_seed{0,1,2}.pt` (from `scripts/train_exploiter.py`), loaded as a `PolicyPlayer` with stochastic acting, no online updates. It is a known counter to `ppo_predicted`, so it tests whether an NPC (in particular MBOM) can adapt to an opponent that already exploits it.
    - **Development exploiter: `exploiter_seed1`.** It is used for tuning only (mixer temperature and similar), together with the train archetypes, and is excluded from the final claims. Final exploiter claims use `exploiter_seed0` and `exploiter_seed2`.
  - [ ] **Final criteria (pre-registered before any final run; `final:` in `configs/mbom.yaml`, `scripts/final_grid.py`)**
    - **Final claims use `exploiter_seed0` and `exploiter_seed2` only**, never `exploiter_seed1` (development; the script refuses it).
    - Design: NPC seeds 0–4, **7 matches per cell** (every opponent), 30 rounds per match, fresh NPC and opponent state per match. Seeds are paired across methods: the same env, opponent and agent seeds for match i of every cell, with a fresh env seed base of 40000.
    - Opponents: the 6 train archetypes, the 2 held-out ones, `naive_learner`, `reasoning_learner` (online, lr3e-4_u64), `exploiter_seed0`, `exploiter_seed2`.
    - Methods: `scripted`, `ppo_none`, `ppo_predicted` (its own online setting), `mbom_no_imagination`, `mbom`, `mbom_random_finetune`, `mbom_single_level1`, `mbom_single_level2`, `mbom_uniform_mix`, and `mbom_top_level`. **`mbom_top_level` is reported as a labeled variant, not as the default.** The `mbom` defaults stay as configured (`cond_input: mix`).
    - **The exploiters were trained against NPC seed 0, so seed 0 and seeds 1–4 are reported separately**, both per exploiter and with the two exploiters pooled. Those four exploiter groups are the claim groups. Every other opponent is reported per opponent (all seeds).
    - **Primary comparisons** (paired per-match win-rate difference, draws = ½):
      1. mbom − `ppo_predicted`
      2. mbom − `no_imagination`
      3. `no_imagination` − `ppo_predicted` (isolates the φ0 finetune schedule)
      4. `uniform_mix` − mbom (value of the mixer)
      5. `single_level1` − mbom (value of the mixer)
      6. `single_level2` − mbom (value of the mixer)
    - **Decision rule, per group:**
      - Paired t-test over matches, Holm–Bonferroni across the 6 primary comparisons within the group.
      - "better" / "worse" if the Holm-adjusted p < 0.05 (sign of the mean); otherwise "no detectable difference".
      - Effects are called practically significant only if |mean| ≥ 0.05 (the go/no-go threshold).
      - The 95% CI is reported with every comparison.
      - Matches within a cell share an NPC checkpoint, so the tests treat matches as independent and the uncertainty is understated (as in the cross-eval). Per-seed results are kept in the CSVs.
    - **Secondary:** paired damage taken per round, and each method's win rate in rounds 1–5 vs the last 5.
    - Saturated cells: win rate ≥ 0.99 for both methods of a comparison is reported as "saturated" in the write-up. The test is still computed.
    - **Amendment 1 (2026-10-18, before any final run). Reason: the time limit (the original design is about 200 process-hours), not results. No final-grid match had been played.** Where it differs from the text above, this amendment applies:
      1. **Matches per cell.** The 8 archetype opponents (6 train + 2 held-out) run at **1 match per cell**: their win rate is saturated and they aren't claim groups. The exploiters (`exploiter_seed0`, `exploiter_seed2`) and the learners stay at **7 matches per cell**.
      2. **Final methods:** `scripted`, `ppo_none`, `ppo_predicted`, `mbom_no_imagination`, `mbom`, and `mbom_uniform_mix` (the mixer control).
         - `mbom_single_level1`, `mbom_single_level2` and `mbom_top_level` are **not run on the final grid**. They are reported from the dev-set runs (`results/mbom/dev_variants.csv`: `exploiter_seed1` + train archetypes) and **labeled as dev-set results** wherever they appear.
         - The primary Holm family becomes the four comparisons that can be computed on the final grid: mbom − `ppo_predicted`, mbom − `no_imagination`, `no_imagination` − `ppo_predicted`, and `uniform_mix` − mbom.
      3. **Conditional on H2.** If H2 passes (MBOM beats `no_imagination` against the lookahead opponent by at least 0.05, paired CI above 0), `mbom_single_level2` and `mbom_random_finetune` are also run on the final grid (`final_grid.py --h2-passed`, which refuses to run unless the saved H2 summary reports SUPPORTED).
         - `single_level2` − mbom then joins the primary Holm family.
         - `random_finetune` − mbom is reported as secondary (unadjusted, outside the family).
         - If H2 fails, neither is run.
      4. **Fallback if compute is still too long:** the exploiter claim groups run on NPC seeds 0, 1, 2 instead of 0–4 (`--reduced-seeds`). Seed 0 stays its own group, and "other seeds" becomes seeds 1–2. If used, this goes into Known limitations (fewer NPC checkpoints per claim group, wider CIs). The learners and archetypes keep seeds 0–4.
  - [ ] **Compute after Amendment 1.** Per method: 40 archetype + 70 learner + 70 exploiter matches = 180 (152 with `--reduced-seeds`). Same per-match rates as below (measured under 10-way load: 4.7 / 8 / 9.6 min per archetype / learner / exploiter match for full-cost MBOM methods, about 0.15 min for the cheap ones, about 0.5 for `random_finetune`).

      | case | process-hours | wall, 10 terminals | wall, 6 terminals |
      |---|---|---|---|
      | H2 fails (6 methods: 2 full-cost × 23.7 h + 4 cheap × 0.45 h) | ≈ 49 | ≈ 5 h | ≈ 6.5–8 h |
      | H2 passes (+ `single_level2` 23.7 h + `random_finetune` 1.5 h) | ≈ 74 | ≈ 7.5 h | ≈ 9.5–12.5 h |
      | H2 fails, `--reduced-seeds` (full-cost 19.2 h each) | ≈ 40 | ≈ 4 h | ≈ 5–6.5 h |
      | H2 passes, `--reduced-seeds` | ≈ 60 | ≈ 6 h | ≈ 7.5–10 h |

      - The 6-terminal range runs from "each process 1.3× faster than under 10-way load" (only the performance cores busy) to "no faster".
      - The learner rate (8 min) is the least certain figure; 2.3–14 min was measured.
      - H2 itself comes first: ≈ 10 process-hours, ≈ 1.5–2 h on 6 terminals.
  - [ ] Compute of the original pre-registered design (superseded by Amendment 1): 10 methods × 12 opponents × 5 seeds × 7 matches = 4,200 matches.
    - Per-match times come from the dev runs on this machine: an i5-14400F with 6 performance and 4 efficiency cores, 10 parallel 1-thread processes, so slower than the single-process benchmark below.
      - Full-cost MBOM methods (`mbom`, `uniform_mix`, `single_level2`, `top_level`): ≈ 4.7 min per archetype match, ≈ 9.6 per exploiter match, ≈ 8 per learner match. The learner figure is uncertain: 2.3–14 min was measured, depending on load.
      - That is **≈ 42 process-hours per full-cost method**, `single_level1` ≈ 25, `random_finetune` ≈ 3.5, and the 4 cheap methods ≈ 1 each.
      - **Total ≈ 200 process-hours, about 20 h wall on 10 processes.**
    - The 8 archetype opponents account for about half of it, and their win rate is saturated.
    - Older single-match estimate, superseded:
    - The 5 imagination methods make up 325 of them, but `random_finetune` skips imagination, so only 4 are expensive.
    - Mean match time: `mbom` / `single_level=2` / `uniform_mix` 38 s at 10 threads (77 s at 1), `single_level=1` 21 s (43 s), `random_finetune` 4 s, `no_imagination` and `ppo_predicted` 2 s. `ppo_none` and `scripted` are also seconds per match.
    - **Total ≈ 2.6 h in one process at 10 threads, or ≈ 0.85 h over 6 processes at 1 thread each** (5.1 h at 1 thread in one process).
    - Caveats: the timed opponents were the archetypes. Learner and exploiter matches may run longer (more steps, so more updates) and should be checked in the first M10 cells. The timing is from one machine.
  - [x] **Final grid results** (user run under Amendment 1, H2 failed, so no `--h2-passed`; full NPC seeds 0–4, no `--reduced-seeds`; `results/m10/final/{status,methods,comparisons}.csv`). All 6 methods have status ok. Win rate is NPC-side, draws = ½, paired over matches.
    - **Claim groups.** The pre-registered text says "four exploiter groups" but describes six seed-split groups: each exploiter at NPC seed 0 and at seeds 1–4, plus both exploiters pooled at each. All six are treated as claim groups.
      - Columns: **Holm** = statistical verdict as pre-registered (Holm-adjusted p < 0.05); **|gain| ≥ 0.05** = the practical-significance check, shown separately. An effect counts as a full claim only if both hold.

      | claim group (n matches) | comparison | paired mean ± 95% CI | p (Holm) | Holm verdict | \|gain\| ≥ 0.05 |
      |---|---|---|---|---|---|
      | both exploiters @ seeds 1–4 (56) | mbom − ppo_predicted | −0.009 ± 0.023 | 0.416 | no detectable difference | no |
      | | mbom − no_imagination | −0.045 ± 0.026 | 0.005 | **worse** | no |
      | | no_imagination − ppo_predicted | +0.035 ± 0.023 | 0.011 | **better** | no |
      | | uniform_mix − mbom | −0.055 ± 0.037 | 0.011 | **worse** | **yes** |
      | both exploiters @ seed 0 (14) | mbom − ppo_predicted | +0.004 ± 0.047 | 1.000 | no detectable difference | no |
      | | mbom − no_imagination | −0.007 ± 0.046 | 1.000 | no detectable difference | no |
      | | no_imagination − ppo_predicted | +0.011 ± 0.027 | 1.000 | no detectable difference | no |
      | | uniform_mix − mbom | +0.060 ± 0.061 | 0.219 | no detectable difference | yes (not significant) |
      | exploiter_seed0 @ seeds 1–4 (28) | mbom − ppo_predicted | −0.008 ± 0.030 | 0.575 | no detectable difference | no |
      | | mbom − no_imagination | −0.052 ± 0.037 | 0.028 | **worse** | **yes** |
      | | no_imagination − ppo_predicted | +0.044 ± 0.031 | 0.028 | **better** | no |
      | | uniform_mix − mbom | −0.065 ± 0.048 | 0.028 | **worse** | **yes** |
      | exploiter_seed0 @ seed 0 (7) | all four | −0.036 … +0.005 | ≥ 0.69 | no detectable difference | no |
      | exploiter_seed2 @ seeds 1–4 (28) | mbom − ppo_predicted | −0.010 ± 0.036 | 0.564 | no detectable difference | no |
      | | mbom − no_imagination | −0.037 ± 0.040 | 0.283 | no detectable difference | no |
      | | no_imagination − ppo_predicted | +0.027 ± 0.036 | 0.360 | no detectable difference | no |
      | | uniform_mix − mbom | −0.045 ± 0.058 | 0.360 | no detectable difference | no |
      | exploiter_seed2 @ seed 0 (7) | mbom − ppo_predicted / − no_imag / no_imag − ppo_pred / uniform − mbom | +0.043 / +0.014 / +0.029 / +0.114 | ≥ 0.17 | no detectable difference | uniform_mix only (not significant) |

    - **Pre-registered verdicts:**
      - **MBOM vs `ppo_predicted`: no detectable difference in any claim group** (largest |mean| 0.043, all Holm p ≥ 0.42).
      - **MBOM vs `no_imagination`: MBOM is worse** on seeds 1–4, pooled (−0.045, statistically but not practically) and for `exploiter_seed0` (−0.052, both). No detectable difference elsewhere.
      - **`no_imagination` vs `ppo_predicted`:** the φ0 finetune schedule is better on seeds 1–4, pooled (+0.035) and for `exploiter_seed0` (+0.044), statistically but not practically. No difference at seed 0.
      - **Mixer value:** `uniform_mix` is worse than mbom on seeds 1–4, pooled (−0.055, statistically and practically) and for `exploiter_seed0` (−0.065, both). The Bayesian mixer beats uniform weights; its job here is to down-weight the poor best-response levels.
      - No method separates at NPC seed 0, where every NPC loses heavily (win rates 0.04–0.44, n = 7–14).
      - `single_level1`, `single_level2` and `top_level` are dev-set results only (`exploiter_seed1`, see M8): `single_level1` −0.202 ± 0.050 vs mbom, `top_level` −0.003 ± 0.012 vs `no_imagination`. `single_level2` wasn't run.
    - **Learners: exploratory, not pre-registered claim groups.** They were reported per opponent only. Same four comparisons, Holm within each opponent:
      - naive: mbom − ppo_predicted −0.052 ± 0.043 (Holm p 0.078; unadjusted 0.020), mbom − no_imagination −0.016 ± 0.049, no_imagination − ppo_predicted −0.036 ± 0.032, uniform_mix − mbom −0.034 ± 0.050. All "no detectable difference".
      - reasoning: −0.024 ± 0.028, −0.025 ± 0.037, +0.000 ± 0.024, and uniform_mix − mbom −0.074 ± 0.055 ("worse", Holm p 0.039, |gain| ≥ 0.05).
      - This matches the dev-set H1 result: no benefit from imagination against adapting learners.
    - **Exploratory comparisons, not pre-registered** (added after the run; unadjusted p; every group in `comparisons.csv`, family `exploratory (not pre-registered)`):

      | group | ppo_predicted − ppo_none | ppo_predicted − scripted |
      |---|---|---|
      | both exploiters @ seed 0 (14) | **−0.311 ± 0.072** (p < 0.001) | −0.199 ± 0.107 (p 0.001) |
      | both exploiters @ seeds 1–4 (56) | +0.046 ± 0.130 (p 0.48) | +0.207 ± 0.085 (p < 0.001) |
      | exploiter_seed0 @ seed 0 / @ seeds 1–4 | −0.364 ± 0.125 / +0.056 ± 0.173 | −0.088 ± 0.099 / +0.231 ± 0.130 |
      | exploiter_seed2 @ seed 0 / @ seeds 1–4 | −0.257 ± 0.085 / +0.037 ± 0.204 | −0.310 ± 0.170 / +0.183 ± 0.117 |
      | naive_learner (35) | +0.325 ± 0.112 | +0.858 ± 0.031 |
      | reasoning_learner (35) | +0.152 ± 0.105 | +0.913 ± 0.031 |

    - **Correction (to M9: "the exploit transfers only partly … against `ppo_none`").** On the NPC checkpoints the exploiters never saw (seeds 1–4), they beat plain PPO about as much as they beat `ppo_predicted`: `ppo_predicted` − `ppo_none` = +0.046 ± 0.130. Per opponent, NPC win rates are `ppo_none` 0.381 vs `ppo_predicted` 0.353 (`exploiter_seed0`) and 0.587 vs 0.565 (`exploiter_seed2`).
      - Only at NPC seed 0, the checkpoint the exploiters were trained against, is `ppo_predicted` much worse than `ppo_none` (−0.311 ± 0.072). The checkpoint-specific part of the exploit targets `ppo_predicted` seed 0.
      - The general part (seeds 1–4) hurts every learned NPC about equally, scripted most.
    - Held-out archetypes (1 match × 5 seeds; `results/m11/heldout_table.csv`): every learned NPC is near saturation.

      | method | kiter: win rate / damage | counter_puncher: win rate / damage |
      |---|---|---|
      | scripted | 0.563 / 2.77 | 0.683 / 2.79 |
      | ppo_none | 0.993 / 0.33 | 0.903 / 1.23 |
      | ppo_predicted | 1.000 / 0.27 | 0.973 / 0.76 |
      | mbom_no_imagination | 1.000 / 0.27 | 0.980 / 0.68 |
      | mbom | 1.000 / 0.24 | 0.947 / 0.90 |
      | mbom_uniform_mix | 0.993 / 0.33 | 0.937 / 0.97 |

    - The train archetypes (1 match × 5 seeds) are saturated: win rate ≥ 0.98 for every learned NPC (`methods.csv`).
  - [x] Seeds: [0, 1, 2, 3, 4]
  - [x] Metrics: win rate, damage taken, rounds 1–5 vs last 5, MBOM prediction accuracy and input drift. Adaptation time after a switch isn't separately measured: the switcher is saturated (≈ 1.00 for every learned NPC).
  - [x] Commit configs and results
- [x] **M11 Plots + tables** (`scripts/make_plots.py --config configs/mbom.yaml` → `results/m11/`, from the saved final grid)
  - [x] Bar charts with 95% CI per claim group: `claim_groups.png` / `.csv`
  - [x] Comparisons table (claim and exploratory groups, labeled): `comparisons_table.csv`
  - [x] Early vs late (rounds 1–5 vs last 5, exploiter groups and learners): `early_late.png` / `.csv`. This replaces the planned "rounds since the switch" curve, because the switcher is saturated.
  - [x] α-weight traces (`mbom`, exploiters and learners, α cap marked): `alpha_traces.png` / `.csv`
  - [x] Generalization table (held-out archetypes): `heldout_table.csv`
  - [x] Commit

### Tier 4 — Demo + human study
- [x] **M12 Game shell + demo panel**
  - [x] Stage 1, game shell (`configs/ui.yaml`, `adaptive_npc/ui/`, retro `render/retro.py`): title, opponent select, HUD, pause, results, `--blind` logging
  - [x] Stage 2, demo panel (Tab, `--panel on`; hidden in `--blind`):
    - [x] Live bars for the predicted next action, with the action taken highlighted and a running accuracy number
    - [x] α over IOP levels (MBOM enemy only, with the 0.79 cap marked)
    - [ ] ~~Posterior over player archetypes~~ (skipped by decision, not built)
  - [x] Commit
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
    - Default: 4 game steps/s, with rendering at 60 fps. Lowered from 8 after play-testing because 8 was too fast to read.
    - Set in `render.yaml`, overridable with `--tick-rate`.
    - Keys: WASD / arrows move, J attack, K block, L dodge, Enter confirm, Esc pause, H legend, F scanlines, **Tab** demo panel, **P** screenshot to `results/demo/`.
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

- **Matches are 30 rounds, a default (Decision 5), not a tuned value, so learners had little time to adapt.** About 600–1,000 learner steps per match is roughly 0.1% of the 1M steps the offline exploiters needed (M9). The NPC's win rate against both learners is still falling at round 30 (`results/mbom/adaptive_test/win_rate_by_round.png`). So the adaptive-opponent results (H1) describe the early phase of adaptation, not where the learners would end up in a longer match.

- **Γ errors compound with k.** After augmentation full-state exactness is 99.4% (1 step) and 98.8% (2 steps) on-policy, 98.9% / 97.3% under random actions. k-step beyond 2 was not measured. The weakest remaining cell is the reward on KO-loss steps (about 86%).

- **The demo MBOM enemy only approximates the evaluated MBOM.** In `play.py` its ~3 s imagination update runs asynchronously in a worker thread, so it takes effect a few steps late, and a trigger that finds the worker busy is skipped. The reported MBOM numbers come from the synchronous evaluation (`ui.mbom_async: false` gives the synchronous demo version).

---

## Verification (every milestone)
- `pytest -q` passes, with new tests added each milestone.
- `python -m scripts.benchmark_env --config configs/env.yaml` reports steps/s for the single and vectorized envs.
- `python -m scripts.balance_matrix --config configs/balance.yaml` writes the matrix and the pass/fail result of the dominance check.
- `python -m scripts.check_learner_drift --config configs/learners.yaml` reports the start→end KL per learner, plus the beat/match/lose outcome per opponent.
- Each training script has a smoke config (`configs/experiments/smoke.yaml`). I run it end to end and check that `results/.../metrics.csv` is written. The user runs the full-budget runs and pastes summaries.
- `python -m scripts.play --tick-rate 8` for manual checks of rendering, controls and the demo panel.
- Commit after each green milestone, for example `M3: game balance matrix + tuned env params`.

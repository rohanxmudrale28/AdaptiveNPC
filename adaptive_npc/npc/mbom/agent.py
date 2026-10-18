"""MBOM agent (Alg. 1, scaled down): frozen ppo_predicted policy fed the mixed IOP prediction; only the IOPs and α
adapt online.

Per real step: the IOP window (same features as the GRU predictor) gives every level's prediction, the mixer's α
combines them into the policy's 8 conditioning dims (cond_input), and after the step the mixer, the real-history
buffer and the window are updated. Every update_every real steps:
  * φ0 is finetuned on the real opponent actions in the buffer
  * for m = 1..top: soft best-response targets on the buffer states by imagination against π∘φ_{m-1} (eq. 5),
    then φ_m is finetuned on them (random_finetune: uniform random action labels instead)

Ablations (method name -> ablation): mbom (none), mbom_no_imagination (φ0 only, α ≡ φ0), mbom_random_finetune (BM),
mbom_single_level1 / mbom_single_level2 (α one-hot at that level, levels up to it built), mbom_uniform_mix (α ≡ 1/M).

Per-round log (MBOMEnv.round_log): α per level (mean over the round and at its end), alpha_cap (α_max is capped at
≈ 0.79 by T = 0.5, M = 3), per-level and mixed prediction accuracy, mixed NLL, policy input drift (entropy and
max-prob of the policy input vs φ0's, fraction of steps where the policy's argmax differs from π(·|φ0)),
updates, update seconds and the Q spread (mean over states of max - min Q over candidate actions).
"""
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.models.env_model import EnvSpec, load_env_model, raw_state
from adaptive_npc.models.player_model import FEAT_DIM, load_predictor, step_features
from adaptive_npc.npc.mbom.imagination import aggregate_q, br_targets, imagined_returns, policy_fns
from adaptive_npc.npc.mbom.iops import IOPSet
from adaptive_npc.npc.mbom.mixer import BayesMixer
from adaptive_npc.npc.ppo_npc import load_policy, model_path

ABLATIONS = ("none", "no_imagination", "random_finetune", "single_level", "uniform_mix")
COND_INPUTS = ("mix", "top_level", "sample_level")


def parse_method(method):
    """'mbom' / 'mbom_<ablation>' / 'mbom_single_level<m>' -> (ablation, level or None)."""
    if method in ("mbom", "mbom_top_level"):  # mbom_top_level: labeled variant, cond_input=top_level (build_mbom)
        return "none", None
    rest = method[len("mbom_"):]
    if rest.startswith("single_level"):
        return "single_level", int(rest[len("single_level"):])
    if rest not in ABLATIONS:
        raise ValueError(f"unknown MBOM method {method!r}")
    return rest, None


def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


class MBOMAgent:
    def __init__(self, mb_cfg, im_cfg, mix_cfg, spec, predictor, window, policy, env_model, ablation="none",
                 level=None, seed=0):
        if ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}, got {ablation!r}")
        self.cond_input = mb_cfg["cond_input"]
        if self.cond_input not in COND_INPUTS:
            raise ValueError(f"cond_input must be one of {COND_INPUTS}, got {self.cond_input!r}")
        M = int(mb_cfg["n_levels"])
        if ablation == "single_level" and not (level is not None and 1 <= level < M):
            raise ValueError(f"single_level needs 1 <= level < {M}, got {level!r}")
        self.spec, self.window, self.policy, self.env_model = spec, int(window), policy, env_model
        self.ablation, self.level, self.M = ablation, level, M
        self.update_every = int(mb_cfg["update_every"])
        self.warmup_grad_steps = int(mb_cfg["warmup_grad_steps"])
        self.im = im_cfg
        self.iops = IOPSet(predictor, M, mb_cfg["lr"], mb_cfg["grad_steps"], mb_cfg["buffer"])
        mode, fixed = {"none": ("bayes", None), "random_finetune": ("bayes", None), "uniform_mix": ("uniform", None),
                       "no_imagination": ("fixed", 0), "single_level": ("fixed", level)}[ablation]
        self.mixer = BayesMixer(M, mix_cfg["decay"], mix_cfg["temperature"], mix_cfg["prior_avg"], mix_cfg["eps"],
                                mix_cfg["prior_floor"], mode=mode, fixed_level=fixed)
        self.top = {"no_imagination": 0, "single_level": level}.get(ablation, M - 1)  # highest level that is built
        self.npc_probs, self.npc_value = policy_fns(policy)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.rng = np.random.default_rng(int(seed))
        self.steps, self.n_updates = 0, 0
        self.round_log = []
        self.reset_round()

    # ------------------------------------------------------------------ real steps
    def reset_round(self):
        self._rows, self.prev, self._pending = [], (None, None), None
        self._acc = {"steps": 0, "alpha": [], "lvl_hit": np.zeros(self.M), "mix_hit": 0, "nll": 0.0,
                     "in_ent": 0.0, "phi0_ent": 0.0, "in_max": 0.0, "phi0_max": 0.0, "changed": 0,
                     "updates": 0, "update_s": 0.0, "q_spread": []}

    def cond(self, obs, raw):
        """Policy conditioning vector [8] for this state (call once per step, before acting)."""
        self._rows.append(step_features(obs, *self.prev))
        win = np.zeros((self.window, FEAT_DIM), dtype=np.float32)
        recent = self._rows[-self.window:]
        win[self.window - len(recent):] = recent
        win = torch.from_numpy(win)
        lp = self.iops.predict_all(win).numpy().astype(np.float64)  # [M, 8]
        alpha = self.mixer.alpha
        if self.cond_input == "mix":
            c = alpha @ lp
        elif self.cond_input == "top_level":
            c = lp[int(np.argmax(alpha))]
        else:
            c = lp[int(self.rng.choice(self.M, p=alpha))]
        c = (c / c.sum()).astype(np.float32)
        x = torch.from_numpy(np.stack([np.concatenate([obs, c]), np.concatenate([obs, lp[0].astype(np.float32)])]))
        pi = self.npc_probs(x).numpy()
        self._pending = {"win": win, "raw": torch.as_tensor(raw, dtype=torch.long), "lp": lp, "c": c,
                         "alpha": alpha, "changed": int(np.argmax(pi[0]) != np.argmax(pi[1]))}
        return c

    def observe(self, a_npc, a_opp):
        p, a = self._pending, int(a_opp)
        self.mixer.update(p["lp"][:, a])
        acc = self._acc
        acc["steps"] += 1
        acc["alpha"].append(p["alpha"])
        acc["lvl_hit"] += p["lp"].argmax(1) == a
        acc["mix_hit"] += int(np.argmax(p["c"]) == a)
        acc["nll"] += -float(np.log(max(p["c"][a], 1e-12)))
        acc["in_ent"] += _entropy(p["c"])
        acc["phi0_ent"] += _entropy(p["lp"][0])
        acc["in_max"] += float(p["c"].max())
        acc["phi0_max"] += float(p["lp"][0].max())
        acc["changed"] += p["changed"]
        self.iops.record(p["win"], p["raw"], a)
        self.prev = (int(a_npc), a)
        self.steps += 1
        if self.steps % self.update_every == 0:
            self.update()

    def end_round(self):
        acc, n = self._acc, max(self._acc["steps"], 1)
        al = np.array(acc["alpha"]) if acc["alpha"] else np.full((1, self.M), np.nan)
        row = {"round": len(self.round_log), "steps": acc["steps"]}
        row.update({f"alpha_l{m}_mean": float(al[:, m].mean()) for m in range(self.M)})
        row.update({f"alpha_l{m}_end": float(self.mixer.alpha[m]) for m in range(self.M)})
        row["alpha_cap"] = self.mixer.alpha_cap if self.mixer.mode == "bayes" else 1.0
        row.update({f"acc_l{m}": float(acc["lvl_hit"][m] / n) for m in range(self.M)})
        row.update({"acc_mixed": acc["mix_hit"] / n, "nll_mixed": acc["nll"] / n,
                    "input_entropy": acc["in_ent"] / n, "phi0_entropy": acc["phi0_ent"] / n,
                    "input_maxprob": acc["in_max"] / n, "phi0_maxprob": acc["phi0_max"] / n,
                    "argmax_changed": acc["changed"] / n, "updates": acc["updates"],
                    "update_seconds": acc["update_s"],
                    "q_spread": float(np.mean(acc["q_spread"])) if acc["q_spread"] else float("nan")})
        self.round_log.append(row)
        self.reset_round()

    # ------------------------------------------------------------------ online update
    def update(self):
        t0 = time.perf_counter()
        wins, raws, acts = self.iops.buffer_tensors()
        self.iops.finetune(0, wins, acts)
        steps = self.warmup_grad_steps if self.n_updates == 0 else None
        for m in range(1, self.top + 1):
            if self.ablation == "random_finetune":
                targets = torch.randint(N_ACTIONS, (len(acts),), generator=self.gen)
            else:
                G = imagined_returns(self.env_model, self.spec, self.npc_probs, self.npc_value,
                                     lambda w, m=m: self.iops.probs(m - 1, w), raws, wins, self.gen,
                                     k=int(self.im["k"]), opp_samples=int(self.im["opp_samples"]),
                                     npc_samples=int(self.im["npc_samples"]), gamma=float(self.im["gamma"]),
                                     imagined_npc=self.im["imagined_npc"], chunk_rows=int(self.im["chunk_rows"]))
                Q = aggregate_q(G, self.im["score_agg"])
                self._acc["q_spread"].append(float((Q.max(1).values - Q.min(1).values).mean()))
                targets = br_targets(Q, self.im["tau_br"])
            self.iops.finetune(m, wins, targets, steps)
        self.n_updates += 1
        self._acc["updates"] += 1
        self._acc["update_s"] += time.perf_counter() - t0


class MBOMEnv(gym.Wrapper):
    """NPCEnv wrapper (like ConditionedEnv): appends the MBOM conditioning vector to the NPC observation."""

    def __init__(self, env, agent):
        super().__init__(env)
        self.agent = agent
        base = env.observation_space
        self.observation_space = spaces.Box(np.concatenate([base.low, np.zeros(N_ACTIONS, np.float32)]),
                                            np.concatenate([base.high, np.ones(N_ACTIONS, np.float32)]),
                                            dtype=np.float32)

    @property
    def round_log(self):
        return self.agent.round_log

    def _cond(self, obs):
        return np.concatenate([obs, self.agent.cond(obs, raw_state(self.env.unwrapped.game))])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.agent.reset_round()
        return self._cond(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.agent.observe(action, info["opp_action"])
        if term or trunc:  # round over: placeholder conditioning, as in ConditionedEnv
            self.agent.end_round()
            return np.concatenate([obs, np.full(N_ACTIONS, 1.0 / N_ACTIONS, np.float32)]), r, term, trunc, info
        return self._cond(obs), r, term, trunc, info


def agent_seed(cfg_seed, npc_seed):
    return int(np.random.SeedSequence([int(cfg_seed), int(npc_seed)]).generate_state(1)[0])


def build_mbom(cfg, env_cfg, method, seed, base_env):
    """MBOM NPC on base_env. Policy = frozen <model_dir>/predicted_seed<seed>.pt. Returns (MBOMEnv, policy)."""
    ablation, level = parse_method(method)
    im = cfg["imagination"]
    model, model_cfg = load_predictor(cfg["predictor_checkpoint"])
    policy = load_policy(model_path(cfg["train"]["model_dir"], "predicted", seed))
    env_model = load_env_model(im["env_model_checkpoint"], bool(im["oracle_model"]), env_cfg)
    mb = {**cfg["mbom"], "cond_input": "top_level"} if method == "mbom_top_level" else cfg["mbom"]
    agent = MBOMAgent(mb, im, cfg["mixer"], EnvSpec(env_cfg), model, int(model_cfg["window"]), policy,
                      env_model, ablation, level, agent_seed(cfg["seed"], seed))
    return MBOMEnv(base_env, agent), policy

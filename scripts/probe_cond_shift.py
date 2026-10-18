"""Distribution-shift probe for M8: how fragile is the frozen ppo_predicted to its conditioning input?

ppo_predicted was trained with raw (frozen) phi0 probabilities appended to the observation; MBOM feeds it a mixture of
IOP levels instead. This plays the frozen policy vs the train archetypes with the 8 conditioning dims replaced by
transformed phi0 outputs (variants in probe.variants) and compares each variant with the raw-phi0 control on the same
env seeds (paired cells). The decision rule (PLAN.md, M8) is checked on probe.decision_variants.

python -m scripts.probe_cond_shift --config configs/mbom.yaml
Writes <probe.out_dir>/metrics.csv (one row per variant x opponent x seed), summary.csv (per variant x opponent),
decision.csv (pooled per variant, paired drop vs control) and the config snapshot + git hash.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.env.observations import OBS_DIM
from adaptive_npc.eval.metrics import aggregate, mean_ci, summarize_rounds
from adaptive_npc.eval.runner import play_match, write_csv
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.models.player_model import OnlinePredictor, load_predictor
from adaptive_npc.npc.ppo_npc import ConditionedEnv, load_policy, model_path
from adaptive_npc.players.scripted import make_player
from adaptive_npc.seeding import seed_everything


def derangement(rng, n=N_ACTIONS):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def make_transform(variant, rng, env):
    """variant dict -> f(p) on phi0 probabilities [8] (float)."""
    kind = variant["kind"]
    if kind == "phi0":
        return lambda p: p
    if kind == "smooth":
        lam = float(variant["lam"])
        return lambda p: (1 - lam) * p + lam / N_ACTIONS
    if kind == "mix_sharp":
        w, temp = float(variant["w"]), float(variant["temp"])

        def f(p):
            s = np.power(np.maximum(p, 1e-12), 1.0 / temp)
            return w * p + (1 - w) * s / s.sum()
        return f
    if kind == "mix_perm":
        w, perm = float(variant["w"]), derangement(rng)
        return lambda p: w * p + (1 - w) * p[perm]
    if kind == "oracle":
        def f(p):
            c = np.zeros(N_ACTIONS)
            c[env.unwrapped.peek_opp_action()] = 1.0
            return c
        return f
    raise ValueError(f"unknown probe variant kind {kind!r}")


class TransformedPredictor:
    """OnlinePredictor proxy for ConditionedEnv: returns transform(phi0) and keeps the raw phi0 output."""

    def __init__(self, base, transform=None):
        self.base, self.transform, self.last_raw = base, transform, None

    def predict(self, obs):
        self.last_raw = self.base.predict(obs)
        return np.asarray(self.transform(self.last_raw.astype(np.float64)), dtype=np.float32)

    def observe(self, a_npc, a_pl):
        self.base.observe(a_npc, a_pl)

    def reset_round(self):
        self.base.reset_round()


def run_cell(cfg, env_cfg, players_cfg, variant, opponent, seed):
    pc = cfg["probe"]
    env_seed = int(pc["env_seed_base"]) + int(seed)
    opp = make_player(opponent, players_cfg, env_cfg, env_seed + 1)
    base_env = NPCEnv(env_cfg, opponent=opp, seed=env_seed)
    model, model_cfg = load_predictor(cfg["predictor_checkpoint"])
    pred = TransformedPredictor(OnlinePredictor(model, int(model_cfg["window"])))  # φ0 frozen, as in training
    env = ConditionedEnv(base_env, "predicted", pred)
    pred.transform = make_transform(variant, np.random.default_rng([int(cfg["seed"]), int(seed)]), env)
    policy = load_policy(model_path(cfg["model_dir"], "predicted", seed))
    det, rng = bool(pc["deterministic"]), np.random.default_rng([int(cfg["seed"]), int(seed), 1])
    stats = {"steps": 0, "changed": 0}

    def act_fn(obs, hist):
        ref = np.concatenate([obs[:OBS_DIM], pred.last_raw.astype(np.float32)])  # same state, raw φ0 input
        with torch.no_grad():
            probs = policy.get_distribution(torch.as_tensor(np.stack([obs, ref]))).distribution.probs.numpy()
        stats["steps"] += 1
        stats["changed"] += int(np.argmax(probs[0]) != np.argmax(probs[1]))
        if det:
            return int(np.argmax(probs[0]))
        p = probs[0].astype(np.float64)
        return int(rng.choice(N_ACTIONS, p=p / p.sum()))

    rounds = play_match(env, act_fn, int(pc["rounds"]), env_seed)
    return rounds, stats["changed"] / max(stats["steps"], 1)


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    pc = cfg["probe"]
    out = Path(args.out or pc["out_dir"])
    save_run_metadata(out, cfg)
    opponents = pc["opponents"] or list(players_cfg["train"])
    rows = []
    for v in pc["variants"]:
        for opp in opponents:
            for seed in pc["seeds"]:
                rounds, changed = run_cell(cfg, env_cfg, players_cfg, v, opp, seed)
                rows.append({"variant": v["name"], "opponent": opp, "seed": int(seed),
                             **summarize_rounds(rounds, env_cfg["hp"], int(pc["rounds"])),
                             "action_change": changed})
            print(f"{v['name']:15s}{opp:12s} done")
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "summary.csv", aggregate(rows, metrics=("win_rate", "damage_taken", "mean_steps", "action_change"),
                                             keys=("variant", "opponent")))

    ctrl = {(r["opponent"], r["seed"]): r for r in rows if r["variant"] == pc["control"]}
    decision, failed = [], []
    for v in pc["variants"]:
        rs = [r for r in rows if r["variant"] == v["name"]]
        wr, wr_ci = mean_ci([r["win_rate"] for r in rs])
        drop, drop_ci = mean_ci([ctrl[(r["opponent"], r["seed"])]["win_rate"] - r["win_rate"] for r in rs])
        is_dec = v["name"] in pc["decision_variants"]
        fails = is_dec and drop > float(pc["max_drop"])
        if fails:
            failed.append(v["name"])
        decision.append({"variant": v["name"], "n_cells": len(rs), "win_rate": wr, "win_rate_ci95": wr_ci,
                         "drop_vs_control": drop, "drop_ci95": drop_ci,
                         "damage_taken": float(np.mean([r["damage_taken"] for r in rs])),
                         "action_change": float(np.mean([r["action_change"] for r in rs])),
                         "decision_variant": is_dec, "fails": fails})
    write_csv(out / "decision.csv", decision)

    print(f"\n{'variant':15s}{'win_rate':>9s}{'drop':>8s}{'+-95%':>8s}{'dmg_taken':>10s}{'act_chg':>9s}")
    for d in decision:
        flag = "  FAIL" if d["fails"] else ("  ok" if d["decision_variant"] else "")
        print(f"{d['variant']:15s}{d['win_rate']:>9.3f}{d['drop_vs_control']:>8.3f}{d['drop_ci95']:>8.3f}"
              f"{d['damage_taken']:>10.2f}{d['action_change']:>9.3f}{flag}")
    verdict = (f"SWITCH cond_input to top_level / sample_level (drop > {pc['max_drop']} on: {', '.join(failed)})"
               if failed else "KEEP cond_input: mix")
    print(f"\ndecision: {verdict}\nwrote results to {out}")


if __name__ == "__main__":
    main()

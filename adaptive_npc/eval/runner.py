"""Evaluation grid: method x opponent x seed -> results/<exp>/<method>/<opponent>/seed_<n>/.

Methods: 'scripted', 'random', 'ppo_none', 'ppo_predicted', 'ppo_oracle', and 'mbom' / 'mbom_<ablation>'
(npc/mbom/agent.py; writes a per-round mbom_rounds.csv). PPO and MBOM methods load <model_dir>/<mode>_seed<seed>.pt. Every cell is one match (persistent opponent state, `rounds` rounds)
played through NPCEnv, so all methods share one code path (and oracle can peek the opponent).
"""
import csv
from pathlib import Path

import numpy as np

from adaptive_npc.config import load_config
from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.eval.metrics import aggregate, summarize_rounds
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.npc.mbom.agent import build_mbom
from adaptive_npc.npc.ppo_npc import ConditionFactory, PolicyNPC, PolicyPlayer, model_path
from adaptive_npc.players.base import RandomPlayer
from adaptive_npc.players.lookahead import LookaheadPlayer
from adaptive_npc.players.naive_learner import NaiveLearner
from adaptive_npc.players.reasoning_learner import ReasoningLearner
from adaptive_npc.players.scripted import make_player


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union in first-seen order (MBOM rows add columns)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def play_match(env, act_fn, n_rounds, seed, on_round_start=None):
    """Play n_rounds rounds in `env`. act_fn(obs, hist) -> action. Returns run_match-style round dicts."""
    out = []
    for rnd in range(n_rounds):
        obs, _ = env.reset(seed=seed if rnd == 0 else None)
        if on_round_start:
            on_round_start()
        hist, ret, done = [], 0.0, False
        while not done:
            a = int(act_fn(obs, hist))
            obs, r, done, _, info = env.step(a)
            hist.append((a, info["opp_action"]))
            ret += r
        out.append({"round": rnd, "winner": info["winner"], "hp_npc": info["hp"][0], "hp_player": info["hp"][1],
                    "steps": len(hist), "ret_npc": ret})
    return out


def build_npc(cfg, env_cfg, players_cfg, method, seed, base_env, npc_seed):
    """NPC side of a match on `base_env` (an NPCEnv with the opponent embedded).
    cfg needs eval.deterministic / eval.predictor_online, online, predictor_checkpoint, train.model_dir.
    Returns (env, act_fn, on_round_start); PPO methods load <model_dir>/<mode>_seed<seed>.pt."""
    ev = cfg["eval"]
    if method.startswith("mbom"):  # needs cfg mbom / imagination / mixer blocks (configs/mbom.yaml)
        env, policy = build_mbom(cfg, env_cfg, method, seed, base_env)
        det = bool(ev["deterministic"])

        def act_fn(obs, hist):
            return int(policy.predict(obs, deterministic=det)[0])
        return env, act_fn, None
    if method.startswith("ppo_"):
        mode = method[len("ppo_"):]
        online = cfg["online"] if ev.get("predictor_online") else None
        env = ConditionFactory(mode, cfg["predictor_checkpoint"], online)(base_env)
        model = PolicyNPC(model_path(cfg["train"]["model_dir"], mode, seed))
        det = bool(ev["deterministic"])

        def act_fn(obs, hist):
            return model.act(obs, det)
        return env, act_fn, None
    if method == "scripted":
        npc = make_player("scripted_npc", players_cfg, env_cfg, npc_seed)
    elif method == "random":
        npc = RandomPlayer(npc_seed)
    else:
        raise ValueError(f"unknown method {method!r}")

    def act_fn(obs, hist):
        return npc.act(obs, hist)
    return base_env, act_fn, npc.reset


def learner_entry(chosen_csv, kind, expected):
    """(lr, update_every) of the chosen online setting for learner `kind` ('naive' / 'reasoning') in chosen.csv;
    fails if the chosen entry is not `expected` (so a re-run of check_learner_drift cannot change it silently)."""
    with open(chosen_csv, encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["learner"] == kind)
    if row["chosen_entry"] != expected:
        raise ValueError(f"{chosen_csv}: chosen entry for {kind} is {row['chosen_entry']!r}, config expects {expected!r}")
    return float(row["lr"]), int(row["update_every"])


def make_opponent(cfg, name, players_cfg, env_cfg, cell_seed, opp_seed):
    """Archetypes via make_player; plus (cfg 'opponents' block):
      exploiter_seed<k>              frozen exploiter <exploiter_dir>/exploiter_seed<k>.pt, stochastic, no updates
      naive_learner / reasoning_learner  pretrained learner (checkpoint seed = cell seed mod len(learner_seeds)),
                                     updating online with the chosen setting from learners_chosen (check_learner_drift)."""
    oc = cfg.get("opponents") or {}
    if name == "lookahead":  # models the cell's ppo_predicted checkpoint (+ pretrained φ0); cfg 'lookahead' block
        return LookaheadPlayer(cfg["lookahead"], env_cfg, model_path(cfg["train"]["model_dir"], "predicted", cell_seed),
                               cfg["predictor_checkpoint"], opp_seed)
    if name.startswith("exploiter_seed"):
        return PolicyPlayer(f"{oc['exploiter_dir']}/{name}.pt", deterministic=False, seed=opp_seed)
    if name in ("naive_learner", "reasoning_learner"):
        kind = name.split("_")[0]
        lr, ue = learner_entry(oc["learners_chosen"], kind, oc["learner_entry"])
        lcfg = load_config(oc["learners_config"])
        lseed = int(oc["learner_seeds"][int(cell_seed) % len(oc["learner_seeds"])])
        ppo, lc = {**lcfg["ppo_lite"], "lr": lr}, lcfg["learners"]
        if kind == "naive":
            return NaiveLearner(lc["naive"]["policy"].format(seed=lseed), ppo, ue, True, opp_seed)
        return ReasoningLearner(lc["reasoning"]["policy"].format(seed=lseed), lc["reasoning"]["npc_model"], ppo, ue,
                                lc["gru_online"], True, opp_seed)
    return make_player(name, players_cfg, env_cfg, opp_seed)


def trace_summary(trace):
    """Step-weighted match means of the MBOM per-round trace (prediction accuracy / NLL, input drift)."""
    w = np.array([r["steps"] for r in trace], dtype=float)
    return {k: float(np.average([r[k] for r in trace], weights=w))
            for k in ("acc_mixed", "nll_mixed", "argmax_changed", "input_entropy", "phi0_entropy")}


def run_cell(cfg, env_cfg, players_cfg, method, opponent, seed):
    ev = cfg["eval"]
    env_seed = int(ev["env_seed_base"]) + int(seed)
    opp = make_opponent(cfg, opponent, players_cfg, env_cfg, seed, env_seed + 1)
    base_env = NPCEnv(env_cfg, opponent=opp, seed=env_seed)
    env, act_fn, on_round_start = build_npc(cfg, env_cfg, players_cfg, method, seed, base_env, env_seed + 2)
    rounds = play_match(env, act_fn, int(ev["rounds"]), env_seed, on_round_start)
    return rounds, getattr(env, "round_log", None)  # per-round MBOM trace (None for other methods)


def run_grid(cfg, env_cfg, players_cfg, out_dir, methods=None, opponents=None, seeds=None):
    ev = cfg["eval"]
    methods = methods or ev["methods"]
    opponents = opponents or ev["opponents"] or (list(players_cfg["train"]) + list(players_cfg["heldout"]))
    seeds = seeds if seeds is not None else ev["seeds"]
    out = Path(out_dir)
    held = set(players_cfg["heldout"])
    rows = []
    for method in methods:
        for opp in opponents:
            for seed in seeds:
                rounds, trace = run_cell(cfg, env_cfg, players_cfg, method, opp, seed)
                cell = out / method / opp / f"seed_{seed}"
                save_run_metadata(cell, cfg)
                write_csv(cell / "rounds.csv", rounds)
                row = {"method": method, "opponent": opp, "heldout": opp in held, "seed": int(seed),
                       **summarize_rounds(rounds, env_cfg["hp"], int(ev["first_n_rounds"]))}
                if trace:
                    write_csv(cell / "mbom_rounds.csv", trace)
                    row.update(trace_summary(trace))
                write_csv(cell / "metrics.csv", [row])
                rows.append(row)
    write_csv(out / "metrics.csv", rows)
    summary = aggregate(rows)
    write_csv(out / "summary.csv", summary)
    return rows, summary

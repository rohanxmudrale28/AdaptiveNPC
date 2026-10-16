"""Evaluation grid: method x opponent x seed -> results/<exp>/<method>/<opponent>/seed_<n>/.

Methods: 'scripted', 'random', 'ppo_none', 'ppo_predicted', 'ppo_oracle'. PPO methods load
<model_dir>/<mode>_seed<seed>.pt. Every cell is one match (persistent opponent state, `rounds` rounds)
played through NPCEnv, so all methods share one code path (and oracle can peek the opponent).
"""
import csv
from pathlib import Path

from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.eval.metrics import aggregate, summarize_rounds
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.npc.ppo_npc import ConditionFactory, PolicyNPC, model_path
from adaptive_npc.players.base import RandomPlayer
from adaptive_npc.players.scripted import make_player


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
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


def run_cell(cfg, env_cfg, players_cfg, method, opponent, seed):
    ev = cfg["eval"]
    env_seed = int(ev["env_seed_base"]) + int(seed)
    opp = make_player(opponent, players_cfg, env_cfg, env_seed + 1)
    base_env = NPCEnv(env_cfg, opponent=opp, seed=env_seed)
    on_round_start = None
    if method.startswith("ppo_"):
        mode = method[len("ppo_"):]
        online = cfg["online"] if ev.get("predictor_online") else None
        env = ConditionFactory(mode, cfg["predictor_checkpoint"], online)(base_env)
        model = PolicyNPC(model_path(cfg["train"]["model_dir"], mode, seed))
        det = bool(ev["deterministic"])

        def act_fn(obs, hist):
            return model.act(obs, det)
    else:
        env = base_env
        if method == "scripted":
            npc = make_player("scripted_npc", players_cfg, env_cfg, env_seed + 2)
        elif method == "random":
            npc = RandomPlayer(env_seed + 2)
        else:
            raise ValueError(f"unknown method {method!r}")
        on_round_start = npc.reset

        def act_fn(obs, hist):
            return npc.act(obs, hist)
    return play_match(env, act_fn, int(ev["rounds"]), env_seed, on_round_start)


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
                rounds = run_cell(cfg, env_cfg, players_cfg, method, opp, seed)
                cell = out / method / opp / f"seed_{seed}"
                save_run_metadata(cell, cfg)
                write_csv(cell / "rounds.csv", rounds)
                row = {"method": method, "opponent": opp, "heldout": opp in held, "seed": int(seed),
                       **summarize_rounds(rounds)}
                write_csv(cell / "metrics.csv", [row])
                rows.append(row)
    write_csv(out / "metrics.csv", rows)
    summary = aggregate(rows, metrics=("win_rate", "mean_return"))
    write_csv(out / "summary.csv", summary)
    return rows, summary

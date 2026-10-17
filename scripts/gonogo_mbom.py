"""M8 go/no-go: MBOM vs ppo_predicted (and vs no_imagination) on the DEVELOPMENT exploiter (exploiter_seed1).

Criterion (PLAN.md, M8; fixed before any run): 5 cells (NPC seeds) x matches_per_cell matches of `rounds` rounds,
fresh NPC state per match, the same env / exploiter seeds per match for every method (paired). GO if MBOM's mean win
rate over the cells beats the control's by >= min_gain and the 95% t-interval of the paired per-match difference is
above 0. Development signal only (exploiter_seed1 is excluded from final claims).

python -m scripts.gonogo_mbom --config configs/mbom.yaml                       # everything, then aggregate
python -m scripts.gonogo_mbom --config configs/mbom.yaml --methods mbom --seeds 0   # one part (parallel processes)
python -m scripts.gonogo_mbom --config configs/mbom.yaml --aggregate-only      # combine finished parts
Per part: <out>/<method>/seed_<n>/{matches.csv, rounds.csv, mbom_rounds.csv (MBOM), config.yaml, git_hash.txt}.
Aggregate: <out>/cells.csv, methods.csv, decision.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.eval.metrics import mean_ci, summarize_rounds
from adaptive_npc.eval.runner import build_npc, make_opponent, play_match, trace_summary, write_csv
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.seeding import seed_everything
from scripts.eval_mbom import prepare


def run_part(cfg, env_cfg, players_cfg, method, seed, out):
    g = cfg["gonogo"]
    cell = out / method / f"seed_{seed}"
    matches, all_rounds, all_trace = [], [], []
    for i in range(int(g["matches_per_cell"])):
        env_seed = int(g["env_seed_base"]) + 100 * int(seed) + i
        c = {**cfg, "seed": int(cfg["seed"]) * 1000 + i}  # fresh imagination rng per match (MBOM agent seed)
        opp = make_opponent(c, g["opponent"], players_cfg, env_cfg, seed, env_seed + 1)
        base = NPCEnv(env_cfg, opponent=opp, seed=env_seed)
        env, act_fn, on_rs = build_npc(c, env_cfg, players_cfg, method, int(seed), base, env_seed + 2)
        rounds = play_match(env, act_fn, int(g["rounds"]), env_seed, on_rs)
        trace = getattr(env, "round_log", None)
        s = summarize_rounds(rounds, env_cfg["hp"], int(g["early_rounds"]))
        late = summarize_rounds(rounds[-int(g["late_rounds"]):], env_cfg["hp"], int(g["late_rounds"]))
        matches.append({"method": method, "seed": int(seed), "match": i, "env_seed": env_seed,
                        "win_rate": s["win_rate"], "win_rate_early": s["win_rate_first_n"],
                        "win_rate_late": late["win_rate"], "damage_taken": s["damage_taken"],
                        "mean_steps": s["mean_steps"], **(trace_summary(trace) if trace else {})})
        all_rounds += [{"match": i, **r} for r in rounds]
        all_trace += [{"match": i, **r} for r in (trace or [])]
        print(f"  {method:22s}seed {seed} match {i}: win rate {s['win_rate']:.3f}")
    save_run_metadata(cell, cfg)
    write_csv(cell / "matches.csv", matches)
    write_csv(cell / "rounds.csv", all_rounds)
    if all_trace:
        write_csv(cell / "mbom_rounds.csv", all_trace)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paired(rows, a, b, key="win_rate"):
    """Per-match differences a - b over (seed, match) present for both methods."""
    idx = {(r["method"], r["seed"], r["match"]): float(r[key]) for r in rows}
    return np.array([v - idx[(b, s, m)] for (meth, s, m), v in idx.items() if meth == a and (b, s, m) in idx])


def aggregate(cfg, out):
    g = cfg["gonogo"]
    rows = []
    for method in g["methods"]:
        for seed in g["seeds"]:
            p = out / method / f"seed_{seed}" / "matches.csv"
            if not p.exists():
                print(f"missing part: {p} -> aggregate skipped")
                return None
            rows += read_csv(p)
    cells = []
    for method in g["methods"]:
        for seed in g["seeds"]:
            rs = [r for r in rows if r["method"] == method and int(r["seed"]) == int(seed)]
            cells.append({"method": method, "seed": int(seed), "matches": len(rs),
                          **{k: float(np.mean([float(r[k]) for r in rs]))
                             for k in ("win_rate", "win_rate_early", "win_rate_late", "damage_taken", "mean_steps")}})
    write_csv(out / "cells.csv", cells)
    methods = []
    for method in g["methods"]:
        cs = [c for c in cells if c["method"] == method]
        wr, wr_ci = mean_ci([c["win_rate"] for c in cs])
        methods.append({"method": method, "cells": len(cs), "win_rate": wr, "win_rate_ci95_over_cells": wr_ci,
                        "win_rate_early": float(np.mean([c["win_rate_early"] for c in cs])),
                        "win_rate_late": float(np.mean([c["win_rate_late"] for c in cs])),
                        "damage_taken": float(np.mean([c["damage_taken"] for c in cs]))})
    write_csv(out / "methods.csv", methods)
    mean_of = {m["method"]: m["win_rate"] for m in methods}
    decision = []
    for ref in (g["control"], g["ablation_ref"]):
        d = paired(rows, "mbom", ref)
        dm, dci = mean_ci(d)
        per_cell = {f"diff_seed{s}": float(np.mean(paired([r for r in rows if int(r["seed"]) == int(s)], "mbom", ref)))
                    for s in g["seeds"]}
        is_ctrl = ref == g["control"]
        gain = mean_of["mbom"] - mean_of[ref]
        go = bool(gain >= float(g["min_gain"]) and dm - dci > 0) if is_ctrl else ""
        decision.append({"comparison": f"mbom - {ref}", "decides": is_ctrl, "mean_gain_over_cells": gain,
                         "paired_mean": dm, "paired_ci95": dci, "paired_ci_low": dm - dci, "n_matches": len(d),
                         **per_cell, "go": go})
    write_csv(out / "decision.csv", decision)

    early, late = f"rounds 1-{g['early_rounds']}", f"last {g['late_rounds']}"
    print(f"\n{'method':22s}{'win_rate':>9s}{'+-95%(cells)':>13s}{early:>11s}{late:>8s}{'dmg':>6s}")
    for m in methods:
        print(f"{m['method']:22s}{m['win_rate']:>9.3f}{m['win_rate_ci95_over_cells']:>13.3f}"
              f"{m['win_rate_early']:>11.3f}{m['win_rate_late']:>8.3f}{m['damage_taken']:>6.2f}")
    print(f"\n{'comparison':34s}{'gain':>7s}{'paired':>8s}{'+-95%':>7s}{'n':>4s}")
    for d in decision:
        print(f"{d['comparison']:34s}{d['mean_gain_over_cells']:>7.3f}{d['paired_mean']:>8.3f}"
              f"{d['paired_ci95']:>7.3f}{d['n_matches']:>4d}")
    ctrl = decision[0]
    print(f"\ndecision (development signal, {g['opponent']}): {'GO' if ctrl['go'] else 'NO-GO'} "
          f"(needs gain >= {g['min_gain']} and paired CI low > 0: gain {ctrl['mean_gain_over_cells']:.3f}, "
          f"CI low {ctrl['paired_ci_low']:.3f})")
    return decision


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = default; 1 for parallel parts)")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cfg = prepare(config_from_args(args))  # records torch_threads in the config snapshot
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    g = cfg["gonogo"]
    out = Path(args.out or g["out_dir"])
    if not args.aggregate_only:
        for method in args.methods or g["methods"]:
            for seed in (args.seeds if args.seeds is not None else g["seeds"]):
                run_part(cfg, env_cfg, players_cfg, method, seed, out)
    aggregate(cfg, out)
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()

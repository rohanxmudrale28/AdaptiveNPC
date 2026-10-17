"""MBOM compute benchmark: seconds per IOP update (with a component breakdown) and per 30-round match, plus a
projection of the M10 grid. Sizes the M10 run; numbers depend on the machine.

python -m scripts.benchmark_mbom --config configs/mbom.yaml
Writes <benchmark.out_dir>/mbom_speed.csv (one row per timed update / match), mbom_m10_projection.csv and the
config snapshot + git hash. Uses the full configs/mbom.yaml settings (buffer, opp_samples, update_every, ...).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.eval.runner import build_npc, play_match, run_cell, write_csv
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.players.scripted import make_player
from adaptive_npc.seeding import seed_everything
from scripts.eval_mbom import prepare


def timed(store, key, fn):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0
        return out
    return wrapper


def bench_update(cfg, env_cfg, players_cfg, threads):
    """Fill the buffer with a real match (no updates), then time full-MBOM updates on it."""
    bc = cfg["benchmark"]
    c = {**cfg, "mbom": {**cfg["mbom"], "update_every": 10 ** 9}}
    env_seed = int(cfg["eval"]["env_seed_base"]) + int(bc["seed"])
    base = NPCEnv(env_cfg, opponent=make_player(bc["opponents"][0], players_cfg, env_cfg, env_seed + 1),
                  seed=env_seed)
    env, act_fn, _ = build_npc(c, env_cfg, players_cfg, "mbom", int(bc["seed"]), base, env_seed + 2)
    play_match(env, act_fn, int(bc["rounds"]), env_seed)
    agent = env.agent
    orig = (agent.env_model.step_t, agent.npc_probs, agent.npc_value, agent.iops.probs)
    rows = []
    for _ in range(int(bc["update_repeats"])):
        parts = {}  # component times inside the update (instance attributes shadow the originals)
        agent.env_model.step_t = timed(parts, "env_model_s", orig[0])
        agent.npc_probs = timed(parts, "policy_s", orig[1])
        agent.npc_value = timed(parts, "policy_s", orig[2])
        agent.iops.probs = timed(parts, "iop_gru_s", orig[3])
        warmup = agent.n_updates == 0  # first update: warmup_grad_steps for φ1..
        t0 = time.perf_counter()
        agent.update()
        total = time.perf_counter() - t0
        rows.append({"kind": "update", "method": "mbom", "opponent": bc["opponents"][0], "threads": threads,
                     "buffer": len(agent.iops.buffer), "warmup": warmup, "steps": "", "updates": 1, "seconds": total,
                     **{k: parts.get(k, 0.0) for k in ("env_model_s", "iop_gru_s", "policy_s")},
                     "other_s": total - sum(parts.values())})
    return rows


def bench_matches(cfg, env_cfg, players_cfg, threads):
    bc = cfg["benchmark"]
    c = {**cfg, "eval": {**cfg["eval"], "rounds": int(bc["rounds"])}}
    rows = []
    for method in bc["methods"]:
        for opp in bc["opponents"]:
            t0 = time.perf_counter()
            rounds, trace = run_cell(c, env_cfg, players_cfg, method, opp, int(bc["seed"]))
            sec = time.perf_counter() - t0
            steps = sum(r["steps"] for r in rounds)
            updates = sum(r["updates"] for r in trace) if trace else 0
            upd_s = sum(r["update_seconds"] for r in trace) if trace else 0.0
            rows.append({"kind": "match", "method": method, "opponent": opp, "threads": threads, "buffer": "",
                         "steps": steps, "updates": updates, "seconds": sec, "update_seconds": upd_s,
                         "sec_per_update": upd_s / updates if updates else ""})
            print(f"  threads={threads:<3d}{method:22s}{opp:16s}{steps:6d} steps {updates:4d} updates {sec:8.1f} s")
    return rows


def project(cfg, rows):
    """M10 projection from the mean match time per method (single process at default threads, and N processes at
    1 thread each)."""
    m10 = cfg["benchmark"]["m10"]
    n_cells = int(m10["n_opponents"]) * int(m10["n_seeds"])
    procs = int(m10["processes"])
    out = []
    for method in m10["methods"]:
        src = m10["same_cost_as"].get(method, method)
        per = {}
        for r in rows:
            if r["kind"] == "match" and r["method"] == src:
                per.setdefault(r["threads"], []).append(r["seconds"])
        if not per:
            continue
        row = {"method": method, "timed_as": src, "matches": n_cells}
        for th, secs in sorted(per.items()):
            label = "default" if th == 0 else str(th)
            row[f"match_s_threads_{label}"] = float(np.mean(secs))
            row[f"total_h_threads_{label}"] = float(np.mean(secs)) * n_cells / 3600
        if 1 in per:
            row[f"total_h_{procs}_procs"] = float(np.mean(per[1])) * n_cells / 3600 / procs
        out.append(row)
    total = {"method": "TOTAL", "timed_as": "", "matches": n_cells * len(out)}
    for k in out[0]:
        if k.startswith("total_h"):
            total[k] = sum(r.get(k, 0.0) for r in out)
    return out + [total]


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = prepare(config_from_args(args))
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    bc = cfg["benchmark"]
    out = Path(args.out or bc["out_dir"])
    default_threads = torch.get_num_threads()
    rows = []
    for threads in bc["threads"]:
        torch.set_num_threads(default_threads if int(threads) == 0 else int(threads))
        print(f"torch threads: {torch.get_num_threads()}")
        rows += bench_update(cfg, env_cfg, players_cfg, int(threads))
        rows += bench_matches(cfg, env_cfg, players_cfg, int(threads))
    torch.set_num_threads(default_threads)
    cfg["benchmark_machine"] = {"default_torch_threads": default_threads}
    save_run_metadata(out / "mbom_speed_meta", cfg)
    write_csv(out / "mbom_speed.csv", rows)
    proj = project(cfg, rows)
    write_csv(out / "mbom_m10_projection.csv", proj)

    print("\nupdate (full MBOM, levels 1-2):")
    for r in rows:
        if r["kind"] == "update":
            print(f"  threads={r['threads']:<3d}{'warm-up ' if r['warmup'] else '        '}{r['seconds']:6.2f} s  (Γ {r['env_model_s']:.2f}, GRU "
                  f"{r['iop_gru_s']:.2f}, policy {r['policy_s']:.2f}, other {r['other_s']:.2f}; buffer {r['buffer']})")
    print("\nM10 projection (hours):")
    for r in proj:
        cols = "  ".join(f"{k}={v:.2f}" for k, v in r.items() if k.startswith("total_h"))
        print(f"  {r['method']:22s}{cols}")
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()

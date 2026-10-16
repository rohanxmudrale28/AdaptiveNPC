"""Steps/s for random vs random: raw core, single gym env, Dummy/Subproc vec envs.

python -m scripts.benchmark_env --config configs/env.yaml
"""
import csv
import time
from pathlib import Path

import numpy as np

from adaptive_npc.config import config_from_args, parse_args
from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.env.vec_env import make_vec_env
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.seeding import seed_everything


def bench_core(cfg, steps, seed):
    rng = np.random.default_rng(seed)
    g = ArenaGame(cfg, rng=np.random.default_rng(seed))
    acts = rng.integers(N_ACTIONS, size=(steps, 2)).tolist()
    t0 = time.perf_counter()
    for a0, a1 in acts:
        _, done = g.step(a0, a1)
        if done:
            g.reset()
    return steps / (time.perf_counter() - t0)


def bench_single(cfg, steps, seed):
    env = NPCEnv(cfg, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)
    acts = rng.integers(N_ACTIONS, size=steps).tolist()
    t0 = time.perf_counter()
    for a in acts:
        _, _, done, _, _ = env.step(a)
        if done:
            env.reset()
    return steps / (time.perf_counter() - t0)


def bench_vec(cfg, steps, seed, n_envs, vec_type):
    venv = make_vec_env(cfg, n_envs=n_envs, vec_env_type=vec_type, seed=seed)
    venv.reset()
    rng = np.random.default_rng(seed)
    n_iter = max(steps // n_envs, 1)
    acts = rng.integers(N_ACTIONS, size=(n_iter, n_envs))
    t0 = time.perf_counter()
    for a in acts:
        venv.step(a)
    rate = n_iter * n_envs / (time.perf_counter() - t0)
    venv.close()
    return rate


def main():
    cfg = config_from_args(parse_args(description=__doc__))
    seed = int(cfg["seed"])
    seed_everything(seed)
    bcfg = cfg["benchmark"]
    steps = int(bcfg["total_steps"])

    rows = [("core", "-", 1, bench_core(cfg, steps, seed)),
            ("single_gym", "-", 1, bench_single(cfg, steps, seed))]
    for vec_type in ("dummy", "subproc"):
        for n in bcfg["n_envs"]:
            rows.append(("vec", vec_type, n, bench_vec(cfg, steps, seed, n, vec_type)))

    out = Path(bcfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setup", "vec_env_type", "n_envs", "steps_per_sec"])
        for setup, vt, n, rate in rows:
            w.writerow([setup, vt, n, f"{rate:.0f}"])
            print(f"{setup:11s} {vt:8s} n_envs={n:<2d} {rate:10.0f} steps/s")
    save_run_metadata(out.parent / "env_speed_run", cfg)
    single = rows[1][3]
    print(f"single env target >= 10000 steps/s: {'PASS' if single >= 10000 else 'FAIL'} ({single:.0f})")


if __name__ == "__main__":
    main()

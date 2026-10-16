"""Train SB3 PPO NPCs (one per conditioning mode x seed) against the train-archetype pool.

python -m scripts.train_npc --config configs/npc_ppo.yaml [--mode predicted] [--seed 0]
Saves <train.model_dir>/<mode>_seed<n>.pt and <train.out_dir>/<mode>/seed_<n>/{config.yaml,git_hash.txt,train_log.csv}.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.env.vec_env import make_vec_env
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.npc.ppo_npc import MODES, ConditionFactory, build_ppo, model_path, save_policy
from adaptive_npc.players.pool import PoolFactory
from adaptive_npc.seeding import seed_everything


class ProgressLog(BaseCallback):
    """Appends (timesteps, ep_rew_mean, ep_len_mean) to a CSV every `every` timesteps."""

    def __init__(self, path, every):
        super().__init__()
        self.path, self.every, self._next = Path(path), int(every), int(every)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timesteps", "ep_rew_mean", "ep_len_mean"])

    def _on_step(self):
        if self.num_timesteps >= self._next and len(self.model.ep_info_buffer) > 0:
            self._next += self.every
            buf = self.model.ep_info_buffer
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([self.num_timesteps, np.mean([e["r"] for e in buf]),
                                        np.mean([e["l"] for e in buf])])
        return True


def train_one(cfg, env_cfg, players_cfg, mode, seed):
    t = cfg["train"]
    seed_everything(seed)
    online = None  # predictor stays frozen during PPO training
    venv = VecMonitor(make_vec_env(
        env_cfg, n_envs=int(t["n_envs"]), vec_env_type=t["vec_env_type"], seed=seed,
        opponent_factory=PoolFactory(t["opponents"] or players_cfg["train"], players_cfg, env_cfg),
        wrapper_factory=ConditionFactory(mode, cfg["predictor_checkpoint"], online)))
    out = Path(t["out_dir"]) / mode / f"seed_{seed}"
    save_run_metadata(out, {**cfg, "mode": mode, "seed": seed})
    model = build_ppo(venv, cfg["ppo"], seed)
    start = time.time()
    model.learn(int(t["total_timesteps"]), callback=ProgressLog(out / "train_log.csv", t["log_every"]))
    path = model_path(t["model_dir"], mode, seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_policy(model, path, cfg["ppo"])
    venv.close()
    sps = int(t["total_timesteps"]) / (time.time() - start)
    print(f"[{mode} seed {seed}] done: {sps:.0f} steps/s -> {path}")


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--mode", choices=MODES, default=None, help="train only this mode")
    args = parser.parse_args()
    cfg = config_from_args(args)
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    modes = [args.mode] if args.mode else cfg["train"]["modes"]
    seeds = [int(cfg["seed"])] if args.seed is not None else cfg["train"]["seeds"]
    for mode in modes:
        for seed in seeds:
            train_one(cfg, env_cfg, players_cfg, mode, int(seed))


if __name__ == "__main__":
    main()

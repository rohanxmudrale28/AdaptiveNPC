"""Evaluate MBOM (and its ablations / baselines) over opponents x seeds through the runner grid.

python -m scripts.eval_mbom --config configs/mbom.yaml [--methods mbom mbom_uniform_mix ppo_predicted]
Writes metrics.csv / summary.csv under eval.out_dir and one <method>/<opponent>/seed_<n>/ folder per cell with the
config snapshot (incl. the evaluation and imagination NPC-action conventions), git hash, rounds.csv, metrics.csv and,
for MBOM methods, mbom_rounds.csv (per-round alpha, per-level IOP accuracy, policy input drift; alpha_max is capped at
~ 0.79 by the mixer temperature, see the alpha_cap column).
"""
import argparse

import torch

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.eval.runner import run_grid
from adaptive_npc.seeding import seed_everything


def prepare(cfg):
    """Fill the keys runner.build_npc expects from configs/mbom.yaml."""
    cfg["online"] = load_config(cfg["predictor_config"])["online"]
    cfg["train"] = {"model_dir": cfg["model_dir"]}
    cfg["conventions"] = {"eval_npc_actions": "greedy" if cfg["eval"]["deterministic"] else "stochastic",
                          "imagined_npc_actions": cfg["imagination"]["imagined_npc"],
                          "torch_threads": torch.get_num_threads()}
    return cfg


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--opponents", nargs="+", default=None)
    args = parser.parse_args()
    cfg = prepare(config_from_args(args))
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    out = args.out or cfg["eval"]["out_dir"]
    _, summary = run_grid(cfg, env_cfg, players_cfg, out, args.methods, args.opponents)
    print(f"{'method':22s}{'opponent':18s}{'win_rate':>10s}{'+-95%CI':>9s}{'dmg_taken':>10s}")
    for r in summary:
        print(f"{r['method']:22s}{r['opponent']:18s}{r['win_rate_mean']:>10.3f}{r['win_rate_ci95']:>9.3f}"
              f"{r['damage_taken_mean']:>10.2f}")
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()

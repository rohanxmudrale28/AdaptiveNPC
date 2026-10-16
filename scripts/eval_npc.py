"""Evaluate NPC methods (scripted / random / PPO variants) over opponents x seeds.

python -m scripts.eval_npc --config configs/npc_ppo.yaml [--methods scripted ppo_none]
Writes metrics.csv (per seed) and summary.csv (mean +- 95% CI) under eval.out_dir, plus one
<method>/<opponent>/seed_<n>/ folder per cell with config snapshot, git hash, rounds.csv, metrics.csv.
"""
import argparse

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.eval.runner import run_grid
from adaptive_npc.seeding import seed_everything


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--opponents", nargs="+", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    cfg["online"] = load_config(cfg["predictor_config"])["online"]
    out = args.out or cfg["eval"]["out_dir"]
    _, summary = run_grid(cfg, env_cfg, players_cfg, out, args.methods, args.opponents)
    print(f"{'method':16s}{'opponent':18s}{'win_rate':>10s}{'+-95%CI':>9s}{'return':>9s}")
    for r in summary:
        print(f"{r['method']:16s}{r['opponent']:18s}{r['win_rate_mean']:>10.3f}{r['win_rate_ci95']:>9.3f}"
              f"{r['mean_return_mean']:>9.2f}")
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()

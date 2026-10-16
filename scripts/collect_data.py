"""Collect predictor training/validation data vs the train archetypes, with a scripted and a random NPC.

python -m scripts.collect_data --config configs/predictor.yaml
"""
import argparse

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.models.predictor_data import collect_split
from adaptive_npc.seeding import seed_everything


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    d = cfg["data"]
    out = args.out or d["dir"]
    train = collect_split(cfg, env_cfg, players_cfg, d["rounds_per_archetype"], d["train_seed_base"], f"{out}/train")
    val = collect_split(cfg, env_cfg, players_cfg, d["val_rounds_per_archetype"], d["val_seed_base"], f"{out}/val")
    save_run_metadata(out, cfg)
    print(f"wrote {len(train)} train and {len(val)} val files to {out}")


if __name__ == "__main__":
    main()

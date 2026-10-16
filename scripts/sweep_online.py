"""Sweep online-finetuning settings (cfg['sweep']) against the frozen GRU.

python -m scripts.sweep_online --config configs/predictor.yaml
Writes online_sweep.csv under eval.out_dir. Per setting, delta = online minus frozen:
  indist  : top-1 on non-mixer train archetypes, all NPCs
  heldout : top-1 on held-out archetypes
  mixer_kl: KL to the true mixer distribution (lower is better)
  dip     : accuracy in the first 75 steps after a switch into camper/repeater/rusher
"""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.eval.predictor_eval import per_archetype_rows, recovery_rows
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.models.player_model import load_predictor
from adaptive_npc.seeding import seed_everything


def summarize(rows, rec, train_names):
    def acc(model, sel):
        return float(np.mean([r["top1_acc"] for r in rows if r["model"] == model and sel(r)]))

    def kl(model):
        return float(np.mean([r["kl_to_true"] for r in rows if r["model"] == model and r["archetype"] == "mixer"]))

    def dip(model):
        v = [r["acc"] for r in rec if r["model"] == model and r["target"] in ("camper", "repeater", "rusher")
             and r["bin_start"] in (0, 25, 50)]
        return float(np.mean(v))

    indist = lambda r: r["archetype"] in train_names and r["archetype"] != "mixer"
    held = lambda r: r["heldout"]
    out = {}
    for m in ("gru_frozen", "gru_online"):
        out[m] = {"indist": acc(m, indist), "heldout": acc(m, held), "mixer_kl": kl(m), "dip": dip(m)}
    return out


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    model, model_cfg = load_predictor(cfg["train"]["checkpoint"])
    cfg["model"] = model_cfg
    out = Path(args.out or cfg["eval"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train_names = set(players_cfg["train"])

    result_rows, ref = [], None
    for s in cfg["sweep"]:
        c = copy.deepcopy(cfg)
        c["online"] = {k: v for k, v in s.items() if k != "name"}
        summ = summarize(per_archetype_rows(env_cfg, players_cfg, c, model),
                         recovery_rows(env_cfg, players_cfg, c, model), train_names)
        ref = ref or summ["gru_frozen"]  # frozen model is identical across settings
        o = summ["gru_online"]
        row = {"name": s["name"], **c["online"]}
        for k in ("indist", "heldout", "mixer_kl", "dip"):
            row[f"d_{k}"] = o[k] - ref[k]
            row[f"online_{k}"] = o[k]
            row[f"frozen_{k}"] = ref[k]
        result_rows.append(row)
        print(f"{s['name']:18s} d_indist {row['d_indist']:+.3f}  d_heldout {row['d_heldout']:+.3f}  "
              f"d_mixer_kl {row['d_mixer_kl']:+.3f}  d_dip {row['d_dip']:+.3f}", flush=True)

    with open(out / "online_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        w.writeheader()
        w.writerows(result_rows)
    save_run_metadata(out / "sweep_meta", cfg)
    print("(delta = online minus frozen; want d_heldout > 0 and d_indist, d_dip close to 0 or positive)")
    print(f"wrote {out / 'online_sweep.csv'}")


if __name__ == "__main__":
    main()

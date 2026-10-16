"""Evaluate the trained predictor: per-archetype top-1 / KL vs baselines, entropy ceiling, recovery curve.

python -m scripts.eval_predictor --config configs/predictor.yaml
Writes metrics_per_archetype.csv, recovery_curve.csv, recovery_curve.png under eval.out_dir.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.eval.predictor_eval import MODELS, per_archetype_rows, recovery_rows
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.models.player_model import load_predictor
from adaptive_npc.seeding import seed_everything

STYLE = {"gru_online": ("#1f77b4", "-"), "gru_frozen": ("#ff7f0e", "-"),
         "majority": ("#7f7f7f", "--"), "repeat_last": ("#2ca02c", ":")}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_table(rows):
    print("\nTop-1 accuracy | KL (NLL, nats; mixer also KL to true dist) | ceiling")
    print(f"{'npc':13s}{'archetype':16s}" + "".join(f"{m:>22s}" for m in MODELS) + "   ceiling")
    keys = sorted({(r["npc"], r["archetype"]) for r in rows})
    for npc, arch in keys:
        sel = {r["model"]: r for r in rows if r["npc"] == npc and r["archetype"] == arch}
        any_r = sel[MODELS[0]]
        cells = ""
        for m in MODELS:
            r = sel[m]
            kl = r["kl_to_true"] if not np.isnan(r["kl_to_true"]) else r["kl_nll"]
            cells += f"{r['top1_acc']:>12.3f} /{kl:7.3f}   "
        tag = " *" if any_r["heldout"] else ""
        print(f"{npc:13s}{arch + tag:16s}{cells}  {any_r['ceiling_acc']:.3f}")
    print("(* = held-out archetype; KL column is KL(true||pred) for the mixer, else NLL = KL(one-hot||pred))")
    mix = next((r for r in rows if r["archetype"] == "mixer"), None)
    if mix:
        print(f"Mixer entropy {mix['entropy_nats']:.3f} nats ({mix['entropy_nats'] / np.log(2):.3f} bits); "
              f"best possible top-1 accuracy = max weight = {mix['ceiling_acc']:.3f} (chance = 0.125)")


def plot_recovery(path, rows, bw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = sorted({r["target"] for r in rows}, key=lambda t: (t == "all", t))
    fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, tgt in zip(axes, targets):
        for m in MODELS:
            pts = sorted((r["bin_start"], r["acc"]) for r in rows if r["target"] == tgt and r["model"] == m)
            x = [b + bw / 2 if b >= 0 else -bw for b, _ in pts]
            c, ls = STYLE[m]
            ax.plot(x, [a for _, a in pts], color=c, ls=ls, lw=2, label=m)
        ax.axvline(0, color="#bbbbbb", lw=1)
        ax.set_title(f"switch into: {tgt}", fontsize=10)
        ax.set_xlabel("steps since switch (-bw = pre-switch)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("top-1 accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    model, model_cfg = load_predictor(cfg["train"]["checkpoint"])
    cfg["model"] = model_cfg
    out = Path(args.out or cfg["eval"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    rows = per_archetype_rows(env_cfg, players_cfg, cfg, model)
    write_csv(out / "metrics_per_archetype.csv", rows)
    print_table(rows)
    rec = recovery_rows(env_cfg, players_cfg, cfg, model)
    write_csv(out / "recovery_curve.csv", rec)
    plot_recovery(out / "recovery_curve.png", rec, int(cfg["eval"]["switch"]["bin_steps"]))
    save_run_metadata(out, cfg)
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()

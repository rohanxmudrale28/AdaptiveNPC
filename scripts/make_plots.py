"""M11 figures and tables from the saved M10 final grid (no new matches).

python -m scripts.make_plots --config configs/mbom.yaml [--out results/m11]
Reads <final.out_dir>/<opponent>/<method>/seed_<n>/{matches.csv, mbom_rounds.csv} and comparisons.csv, writes:
  claim_groups.png / .csv      NPC win rate per method with 95% CI (over matches) in each pre-registered claim group
  comparisons_table.csv        the claim-group and exploratory-group comparisons (from comparisons.csv), labeled
  early_late.png / .csv        win rate in rounds 1-5 vs the last 5, per method (exploiter groups, learners)
  alpha_traces.png / .csv      MBOM mixer weights alpha per level over the 30 rounds (mean over matches), alpha cap marked
  heldout_table.csv            held-out archetypes (kiter, counter_puncher): win rate, damage, steps per method
Win rate is NPC-side, draws = 1/2. Colors: dataviz reference palette, categorical slots in fixed order (validated);
every bar / point is also labeled by method on the axis, and every figure has its CSV as the table view.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_npc.config import add_config_args, config_from_args  # noqa: E402
from adaptive_npc.eval.metrics import mean_ci  # noqa: E402
from adaptive_npc.eval.runner import write_csv  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {"scripted": "scripted", "ppo_none": "PPO", "ppo_predicted": "pred-PPO", "mbom_no_imagination": "no imag.",
         "mbom": "MBOM", "mbom_uniform_mix": "uniform mix"}


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style():
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK})


def frame(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_matches(fc, methods):
    root, rows = Path(fc["out_dir"]), []
    for m in methods:
        for opp in fc["opponents"]:
            for p in sorted((root / opp / m).glob("seed_*/matches.csv")):
                rows += [{**r, "opponent": opp, "method": m} for r in read(p)]
    return rows


def claim_groups(fc):
    tr = str(fc["exploiter_train_seed"])
    ex = fc["exploiters"]
    out = []
    for name, opps in [(e, [e]) for e in ex] + [("both exploiters", ex)]:
        out.append((f"{name} @ NPC seed {tr}", lambda r, o=opps: r["opponent"] in o and r["seed"] == tr))
        out.append((f"{name} @ NPC seeds 1-4", lambda r, o=opps: r["opponent"] in o and r["seed"] != tr))
    return out


def fig_claim_groups(fc, methods, rows, out):
    groups, table = claim_groups(fc), []
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharey=True, facecolor=SURFACE)
    x = np.arange(len(methods))
    for ax, (gname, sel) in zip(axes.flat, groups):
        frame(ax)
        for i, m in enumerate(methods):
            v = [float(r["win_rate"]) for r in rows if r["method"] == m and sel(r)]
            mean, ci = mean_ci(v)
            table.append({"group": gname, "method": m, "n_matches": len(v), "win_rate": mean, "ci95": ci})
            ax.bar(i, mean, width=0.68, color=SLOTS[i], edgecolor=SURFACE, linewidth=2)
            ax.errorbar(i, mean, yerr=ci, color=INK2, capsize=3, linewidth=1.2)
        n = table[-1]["n_matches"]
        ax.set_title(f"{gname}  (n = {n} matches per bar)", loc="left", fontsize=10)
        ax.set_xticks(x, [SHORT[m] for m in methods], rotation=0, fontsize=8.5)
        ax.set_ylim(0, 1)
    for ax in axes[:, 0]:
        ax.set_ylabel("NPC win rate (draws = ½)")
    fig.suptitle("Final grid, pre-registered claim groups: NPC win rate vs the exploiters, mean ± 95% CI over matches",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "claim_groups.png", dpi=150, facecolor=SURFACE)
    write_csv(out / "claim_groups.csv", table)


def fig_early_late(fc, methods, rows, out):
    tr = str(fc["exploiter_train_seed"])
    ex = set(fc["exploiters"])
    panels = [(f"exploiters @ NPC seed {tr}", lambda r: r["opponent"] in ex and r["seed"] == tr),
              ("exploiters @ NPC seeds 1-4", lambda r: r["opponent"] in ex and r["seed"] != tr),
              ("naive learner (exploratory)", lambda r: r["opponent"] == "naive_learner"),
              ("reasoning learner (exploratory)", lambda r: r["opponent"] == "reasoning_learner")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    table = []
    for ax, (gname, sel) in zip(axes, panels):
        frame(ax)
        for i, m in enumerate(methods):
            rs = [r for r in rows if r["method"] == m and sel(r)]
            (e, eci), (l, lci) = (mean_ci([float(r[k]) for r in rs]) for k in ("win_rate_early", "win_rate_late"))
            table.append({"group": gname, "method": m, "n_matches": len(rs), "rounds_1_5": e, "rounds_1_5_ci95": eci,
                          "last_5": l, "last_5_ci95": lci})
            ax.plot([i - 0.15, i + 0.15], [e, l], color=SLOTS[i], linewidth=2)
            ax.errorbar(i - 0.15, e, yerr=eci, fmt="o", color=SLOTS[i], markersize=6, markerfacecolor=SURFACE,
                        markeredgewidth=1.8, capsize=2, elinewidth=1)
            ax.errorbar(i + 0.15, l, yerr=lci, fmt="o", color=SLOTS[i], markersize=6, capsize=2, elinewidth=1)
        ax.set_title(gname, loc="left", fontsize=10)
        ax.set_xticks(range(len(methods)), [SHORT[m] for m in methods], rotation=45, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("NPC win rate (draws = ½)")
    fig.suptitle("Rounds 1-5 (open circle) vs last 5 rounds (filled), mean ± 95% CI over matches",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "early_late.png", dpi=150, facecolor=SURFACE)
    write_csv(out / "early_late.csv", table)


def fig_alpha(fc, mix_cfg, out):
    root = Path(fc["out_dir"])
    opps = list(fc["exploiters"]) + ["naive_learner", "reasoning_learner"]
    T, M = float(mix_cfg["temperature"]), 3
    cap = float(np.exp(1 / T) / (np.exp(1 / T) + M - 1))
    fig, axes = plt.subplots(1, len(opps), figsize=(13.5, 3.8), sharey=True, facecolor=SURFACE)
    table, x = [], np.arange(1, int(fc["rounds"]) + 1)
    for ax, opp in zip(axes, opps):
        frame(ax)
        trace = []
        for p in sorted((root / opp / "mbom").glob("seed_*/mbom_rounds.csv")):
            trace += read(p)
        ends = []
        for lvl, color in zip(range(M), SLOTS[:M]):
            y = [np.mean([float(r[f"alpha_l{lvl}_mean"]) for r in trace if int(r["round"]) == k - 1]) for k in x]
            ax.plot(x, y, color=color, linewidth=2, label=f"φ{lvl}")
            ends.append(y[-1])
            table += [{"opponent": opp, "level": lvl, "round": int(k), "alpha_mean": float(v)} for k, v in zip(x, y)]
        # direct labels: φ0 just under the cap line; φ1 and φ2 coincide, so one shared label
        ax.annotate("φ0", (x[-1], ends[0]), xytext=(4, -9), textcoords="offset points", color=INK2, fontsize=8.5,
                    va="center", annotation_clip=False)
        ax.annotate("φ1, φ2", (x[-1], np.mean(ends[1:])), xytext=(4, 0), textcoords="offset points", color=INK2,
                    fontsize=8.5, va="center", annotation_clip=False)
        ax.axhline(cap, color=INK2, linestyle="--", linewidth=1)
        ax.set_title(opp.replace("_", " "), loc="left", fontsize=10)
        ax.set_xlabel("round of the match")
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlim(0.5, x[-1] + 0.5)
    fig.subplots_adjust(wspace=0.3)
    axes[0].text(1, cap + 0.02, f"α cap ≈ {cap:.2f} (T = {T})", color=INK2, fontsize=8)
    axes[0].set_ylabel("mixer weight α (mean over matches)")
    axes[0].set_ylim(0, 0.9)
    axes[0].legend(loc="center right", frameon=False, fontsize=8.5)
    n = len(trace) // int(fc["rounds"])
    fig.suptitle(f"MBOM mixer weights per round, final grid ({n} matches per panel)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.98, 0.92))
    fig.savefig(out / "alpha_traces.png", dpi=150, facecolor=SURFACE)
    write_csv(out / "alpha_traces.csv", table)


def heldout_table(fc, methods, rows, players_heldout, out):
    table = []
    for opp in players_heldout:
        for m in methods:
            rs = [r for r in rows if r["opponent"] == opp and r["method"] == m]
            wr, ci = mean_ci([float(r["win_rate"]) for r in rs])
            table.append({"opponent": opp, "method": m, "n_matches": len(rs), "win_rate": wr, "win_rate_ci95": ci,
                          "damage_taken": float(np.mean([float(r["damage_taken"]) for r in rs])),
                          "mean_steps": float(np.mean([float(r["mean_steps"]) for r in rs]))})
    write_csv(out / "heldout_table.csv", table)
    return table


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    fc = cfg["final"]
    out = Path(args.out or "results/m11")
    out.mkdir(parents=True, exist_ok=True)
    style()
    methods = list(fc["methods"])
    rows = load_matches(fc, methods)
    fig_claim_groups(fc, methods, rows, out)
    fig_early_late(fc, methods, rows, out)
    fig_alpha(fc, cfg["mixer"], out)
    comps = [r for r in read(Path(fc["out_dir"]) / "comparisons.csv") if not r["group_type"].startswith("per-opponent")]
    write_csv(out / "comparisons_table.csv", comps)
    from adaptive_npc.config import load_config
    ho = heldout_table(fc, methods, rows, load_config(cfg["players_config"])["heldout"], out)
    for r in ho:
        print(f"{r['opponent']:16s}{r['method']:22s}{r['win_rate']:.3f} ± {r['win_rate_ci95']:.3f}  "
              f"dmg {r['damage_taken']:.2f}  steps {r['mean_steps']:.1f}  (n={r['n_matches']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

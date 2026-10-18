"""NPC win rate per round (averaged over matches) against the learners, from the saved adaptive-test runs.

python -m scripts.plot_adaptive_rounds --config configs/mbom.yaml
Reads <adaptive_test.out_dir>/<opponent>/<method>/seed_<n>/rounds.csv (no new matches) and writes
win_rate_by_round.csv (opponent, method, round, n matches, win rate, 95% CI; draws = 1/2) and win_rate_by_round.png.
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

OPPONENTS = ("naive_learner", "reasoning_learner")
# dataviz reference palette, categorical slots 1-3 in fixed order (validated: CVD ΔE 9.2; aqua < 3:1 contrast ->
# direct labels + the CSV as table view); markers are a second encoding
STYLE = {"mbom": ("#2a78d6", "o", "MBOM"), "mbom_no_imagination": ("#eb6834", "s", "no imagination"),
         "ppo_predicted": ("#1baf7a", "^", "ppo_predicted")}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def win_score(w):
    return 1.0 if w == 0 else 0.5 if w == -1 else 0.0


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    at = config_from_args(args)["adaptive_test"]
    root = Path(args.out or at["out_dir"])
    n_rounds = int(at["rounds"])
    rows = []
    for opp in OPPONENTS:
        for method in at["methods"]:
            per_round = [[] for _ in range(n_rounds)]
            for seed in at["seeds"]:
                with open(root / opp / method / f"seed_{seed}" / "rounds.csv", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        per_round[int(r["round"])].append(win_score(int(r["winner"])))
            for rnd, vals in enumerate(per_round):
                m, ci = mean_ci(vals)
                rows.append({"opponent": opp, "method": method, "round": rnd + 1, "n_matches": len(vals),
                             "win_rate": m, "ci95": ci})
    write_csv(root / "win_rate_by_round.csv", rows)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    x = np.arange(1, n_rounds + 1)
    for ax, opp in zip(axes, OPPONENTS):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ends = []
        for method in at["methods"]:
            color, marker, label = STYLE[method]
            y = np.array([r["win_rate"] for r in rows if r["opponent"] == opp and r["method"] == method])
            ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=4.5, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
            ends.append((y[-1], label))
        # direct labels at the line ends, nudged apart so they do not collide
        placed = []
        for yv, label in sorted(ends):
            yl = max([yv] + [p + 0.035 for p in placed])
            placed.append(yl)
            ax.annotate(label, (n_rounds, yv), xytext=(n_rounds + 0.8, yl), color=INK2, fontsize=8.5, va="center",
                        annotation_clip=False)
        ax.set_title(opp.replace("_", " "), color=INK, fontsize=11, loc="left")
        ax.set_xlabel("round of the match")
        ax.set_xlim(0.5, n_rounds + 0.5)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    axes[0].set_ylabel("NPC win rate (draws = ½)")
    axes[0].set_ylim(0.5, 1.02)
    n = rows[0]["n_matches"]
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(f"NPC win rate per round vs online learners ({n} matches per point: 5 NPC seeds × 7)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    fig.savefig(root / "win_rate_by_round.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {root / 'win_rate_by_round.csv'} and .png")


if __name__ == "__main__":
    main()

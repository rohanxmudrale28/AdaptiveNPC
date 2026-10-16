"""Round-robin balance matrix over all archetypes + checks.

python -m scripts.balance_matrix --config configs/balance.yaml [--note "text for log.md"]
Exit code 0 if all checks pass, 1 otherwise. Draw-rate flags are warnings only.
"""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.eval.balance import build_matrices, play_pair, run_checks
from adaptive_npc.logging_utils import save_run_metadata


def write_matrix(path, names, mat):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_vs_col"] + names)
        for n, row in zip(names, mat):
            w.writerow([n] + [f"{v:.4f}" for v in row])


def heatmap(path, names, mat, title, cmap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def text_matrix(title, names, mat):
    lines = [title, " " * 16 + "".join(f"{x[:9]:>10s}" for x in names)]
    for nm, row in zip(names, mat):
        lines.append(f"{nm:16s}" + "".join(f"{v:10.3f}" for v in row))
    return lines


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--note", default=None, help="label appended to log.md with this run")
    args = parser.parse_args()
    cfg = config_from_args(args)
    env_cfg = load_config(cfg["env_config"])
    players_cfg = load_config(cfg["players_config"])
    out = Path(args.out or cfg["out"])
    out.mkdir(parents=True, exist_ok=True)

    names = list(players_cfg["train"]) + list(players_cfg["heldout"]) + list(cfg["extra_players"])
    n = len(names)
    tasks, index = [], []
    for i in range(n):
        for j in range(n):
            for s in cfg["seeds"]:
                tasks.append((env_cfg, players_cfg, names[i], names[j], int(s) * 7919 + i * 101 + j,
                              int(cfg["rounds_per_pair"])))
                index.append((i, j))
    print(f"{n} players, {len(tasks)} matches of {cfg['rounds_per_pair']} rounds, "
          f"{cfg['workers']} workers", flush=True)
    with Pool(int(cfg["workers"])) as pool:
        results = pool.map(play_pair, tasks, chunksize=1)

    counts = np.zeros((n, n, 3))
    for (i, j), r in zip(index, results):
        counts[i, j] += r
    slot0, winrate, draw, decisive = build_matrices(counts)

    write_matrix(out / "winrate_matrix.csv", names, winrate)
    write_matrix(out / "slot0_matrix.csv", names, slot0)
    write_matrix(out / "draw_rate_matrix.csv", names, draw)
    write_matrix(out / "decisive_matrix.csv", names, decisive)
    heatmap(out / "winrate_heatmap.png", names, winrate,
            "Win rate of row vs column (draws = 1/2, both sides)", "RdBu")
    heatmap(out / "draw_rate_heatmap.png", names, draw, "Draw rate per pair", "viridis")
    save_run_metadata(out / "run", {"balance": cfg, "env": env_cfg, "players": players_cfg})

    checks, flagged = run_checks(names, slot0, winrate, draw, decisive, cfg)
    lines = text_matrix("Win-rate matrix (row vs col, draws = 1/2, both sides):", names, winrate)
    lines += [""] + text_matrix("Decisive win-rate matrix (wins / (wins + losses), row vs col):",
                                names, decisive)
    lines += [""] + text_matrix("Draw-rate matrix (unordered pair):", names, draw)
    lines += ["", f"Pairs with draw rate > {cfg['draw_flag_threshold']:.0%}: "
              + (", ".join(f"{a} vs {b} ({d:.1%})" for a, b, d in flagged) or "none"),
              "", "Checks:"]
    for name, ok, detail in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    all_ok = all(ok for _, ok, _ in checks)
    lines.append(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    report = "\n".join(lines)
    print(report)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    with open(out / "log.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Iteration: {args.note or '(no note)'}\n\n```\n{report}\n```\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

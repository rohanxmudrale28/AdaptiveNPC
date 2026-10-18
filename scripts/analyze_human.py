"""Analyse the human-evaluation logs (M13): results/human/<ID>_session.json from scripts/human_eval.py.

python -m scripts.analyze_human
python -m scripts.analyze_human --dir results/human --out-dir results/human
Writes human_summary.csv, human_curve.csv, human_summary.txt and human_eval.png into the output directory.
Win rate = rounds the human won / rounds played (draws count as not won). Only completed test matches are used.
EXPLORATORY: n is small, so intervals are wide and nothing here is a confirmed result.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from adaptive_npc.config import load_config

NOTE = ("NOTE: n is small and this analysis is exploratory. Intervals are 95% t-intervals across participants "
        "and are wide; do not read these as confirmed effects.")


def load_sessions(directory):
    """One dict per participant: {id, order, npcs: {npc: {matches: [round-lists in play order], ratings: [...]}}}."""
    out = []
    for f in sorted(Path(directory).glob("*_session.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        per_npc = {}
        for m in s["matches"]:
            if m.get("phase") != "test" or not m.get("completed") or not m.get("summary"):
                continue
            rec = per_npc.setdefault(m["opponent"], {"matches": [], "ratings": []})
            rec["matches"].append(m["summary"]["rounds"])
            if m.get("difficulty_rating") is not None:
                rec["ratings"].append(m["difficulty_rating"])
        out.append({"id": s["participant"], "order": s.get("order", {}), "npcs": per_npc})
    return out


def wins(rounds):
    return np.array([1.0 if r["winner"] == "human" else 0.0 for r in rounds])


def block(p, npc):
    """All rounds against one NPC in play order (matches concatenated), or empty."""
    rec = p["npcs"].get(npc)
    return [r for m in rec["matches"] for r in m] if rec else []


def pred_acc(p, npc):
    """Mean of the per-round prediction accuracies (NaN for NPCs that do not predict)."""
    vals = [r["pred_accuracy"] for r in block(p, npc) if r.get("pred_accuracy") is not None]
    return float(np.mean(vals)) if vals else np.nan


def slope(w):
    """OLS slope of win (0/1) on round index within the NPC block: change in win probability per round."""
    return float(np.polyfit(np.arange(len(w)), w, 1)[0]) if len(w) >= 3 else np.nan


def match_gain(p, npc):
    """Win rate in the last match against the NPC minus the first."""
    ms = p["npcs"].get(npc, {}).get("matches", [])
    return float(wins(ms[-1]).mean() - wins(ms[0]).mean()) if len(ms) >= 2 else np.nan


def mean_ci(x, alpha=0.05):
    """(mean, lo, hi, n) over the finite values; t-interval; NaN bounds when n < 2."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, n
    h = float(stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def per_round_curve(sessions, npc):
    """Mean human win rate per round index within the NPC block, across participants. Returns (mean, n_per_round)."""
    ws = [wins(block(p, npc)) for p in sessions if block(p, npc)]
    if not ws:
        return np.array([]), np.array([])
    L = max(len(w) for w in ws)
    mat = np.full((len(ws), L), np.nan)
    for i, w in enumerate(ws):
        mat[i, :len(w)] = w
    return np.nanmean(mat, axis=0), np.sum(np.isfinite(mat), axis=0)


def analyze(sessions, npcs=("scripted", "ppo_predicted"), alpha=0.05):
    """Return (rows, text). rows = list of dicts for the summary CSV."""
    a, b = npcs
    rows, lines = [], []

    def add(metric, values, scope):
        m, lo, hi, n = mean_ci(values, alpha)
        rows.append({"metric": metric, "scope": scope, "mean": m, "ci_lo": lo, "ci_hi": hi, "n": n})
        if not np.isfinite(m):
            lines.append(f"  {metric:48s} n/a")
            return
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "[n/a, need n >= 2]"
        lines.append(f"  {metric:48s} {m:+.3f}  95% CI {ci}  (n={n})")

    lines += [NOTE, "", f"participants: {len(sessions)} ({', '.join(p['id'] for p in sessions)})", ""]
    lines.append("Per participant: human win rate (and NPC prediction accuracy), completed test matches only")
    for p in sessions:
        cells = []
        for n in npcs:
            w = wins(block(p, n))
            acc = pred_acc(p, n)
            cells.append("no data" if not len(w) else
                         f"{n}: win {w.mean():.2f} ({int(w.sum())}/{len(w)})" + (f", acc {acc:.2f}" if np.isfinite(acc) else ""))
        lines.append(f"  {p['id']:10s} A={p['order'].get('A', '?'):14s} " + " | ".join(cells))
        rows.append({"metric": "participant", "scope": p["id"],
                     **{f"win_{n}": (float(wins(block(p, n)).mean()) if block(p, n) else np.nan) for n in npcs},
                     **{f"acc_{n}": pred_acc(p, n) for n in npcs}})

    lines += ["", "Win rates (pooled over all rounds, then the mean of per-participant win rates)"]
    for n in npcs:
        w = np.concatenate([wins(block(p, n)) for p in sessions if block(p, n)] or [np.array([])])
        if len(w):
            lines.append(f"  {n:14s} pooled win rate {w.mean():.3f} ({int(w.sum())}/{len(w)} rounds)")
        add(f"win rate vs {n} (mean over participants)", [wins(block(p, n)).mean() for p in sessions if block(p, n)], n)

    lines += ["", "NPC prediction accuracy on humans (the scripted NPC does not predict, so it has none)"]
    for n in npcs:
        add(f"prediction accuracy of {n}", [pred_acc(p, n) for p in sessions], n)

    lines += ["", "Learning within an opponent: second-match minus first-match win rate, and win-probability slope per round"]
    for n in npcs:
        add(f"win-rate gain, match 2 - match 1, vs {n}", [match_gain(p, n) for p in sessions], n)
    for n in npcs:
        add(f"win-probability slope per round vs {n}", [slope(wins(block(p, n))) for p in sessions], n)

    lines += ["", f"Difficulty ratings (1-5, optional)"]
    for n in npcs:
        add(f"difficulty rating of {n}", [np.mean(p["npcs"][n]["ratings"]) for p in sessions
                                          if n in p["npcs"] and p["npcs"][n]["ratings"]], n)

    both = [p for p in sessions if block(p, a) and block(p, b)]
    lines += ["", f"Paired differences ({a} minus {b}), one pair per participant who played both (n={len(both)})"]
    add(f"win rate: {a} - {b}", [wins(block(p, a)).mean() - wins(block(p, b)).mean() for p in both], "paired")
    add(f"win-rate gain: {a} - {b}", [match_gain(p, a) - match_gain(p, b) for p in both], "paired")
    add(f"slope per round: {a} - {b}", [slope(wins(block(p, a))) - slope(wins(block(p, b))) for p in both], "paired")
    add(f"difficulty rating: {a} - {b}", [np.mean(p["npcs"][a]["ratings"]) - np.mean(p["npcs"][b]["ratings"]) for p in both
                                         if p["npcs"][a]["ratings"] and p["npcs"][b]["ratings"]], "paired")

    sa = mean_ci([slope(wins(block(p, a))) for p in sessions], alpha)
    sb = mean_ci([slope(wins(block(p, b))) for p in sessions], alpha)
    lines += ["", f"Does the human's win rate rise against {a} but not against {b}?",
              f"  mean slope vs {a}: {sa[0]:+.4f} per round, vs {b}: {sb[0]:+.4f} per round "
              f"(CIs in the rows above; the per-round curve is in human_curve.csv and the figure)", "", NOTE]
    return rows, "\n".join(lines)


def plot(sessions, npcs, path, window=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = dict(zip(npcs, ("#1f77b4", "#d95f02")))
    ax = axes[0]
    for n in npcs:
        for p in sessions:  # thin lines: each participant's running win rate
            w = wins(block(p, n))
            if len(w):
                ax.plot(np.arange(1, len(w) + 1), w.cumsum() / np.arange(1, len(w) + 1), color=colors[n], alpha=0.2, lw=0.8)
        c, _ = per_round_curve(sessions, n)
        if len(c):
            k = max(1, min(window, len(c)))
            sm = np.convolve(np.nan_to_num(c, nan=float(np.nanmean(c))), np.ones(k) / k, mode="valid")
            ax.plot(np.arange(1, len(sm) + 1) + (k - 1) / 2, sm, color=colors[n], lw=2.2, label=f"vs {n} (mean, {k}-round avg)")
    ax.set(xlabel="round within opponent (its matches concatenated)", ylabel="human win rate",
           title="Human win rate by round (thin: running rate per participant)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for i, n in enumerate(npcs):
        vals = [wins(block(p, n)).mean() for p in sessions if block(p, n)]
        m, lo, hi, _ = mean_ci(vals)
        ax.scatter(i + np.linspace(-0.08, 0.08, len(vals)), vals, color=colors[n], alpha=0.5, s=18)
        if np.isfinite(m):
            ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None, fmt="D", color="k", capsize=5)
    ax.set_xticks(range(len(npcs)), list(npcs))
    ax.set(ylabel="human win rate (dots: participants; diamond: mean, 95% CI)", title="Win rate by opponent")
    ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Human evaluation: exploratory, n is small", fontsize=10, color="firebrick")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/human_eval.yaml")
    p.add_argument("--dir", default=None, help="folder with <ID>_session.json files (default: out_dir of the config)")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    src = Path(args.dir or cfg["out_dir"])
    out = Path(args.out_dir or src)
    sessions = load_sessions(src)
    if not sessions:
        raise SystemExit(f"no *_session.json files in {src}. Run: python -m scripts.human_eval --participant P01")
    npcs = tuple(cfg["npcs"])
    rows, text = analyze(sessions, npcs, cfg["analysis"]["alpha"])
    out.mkdir(parents=True, exist_ok=True)
    head = ["metric", "scope", "mean", "ci_lo", "ci_hi", "n"]
    keys = head + sorted({k for r in rows for k in r} - set(head))
    with open(out / "human_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, restval="")
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "human_curve.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["npc", "round", "human_win_rate", "n_participants"])
        for n in npcs:
            c, cnt = per_round_curve(sessions, n)
            wr.writerows([[n, i + 1, f"{c[i]:.4f}", int(cnt[i])] for i in range(len(c))])
    (out / "human_summary.txt").write_text(text + "\n", encoding="utf-8")
    fig = plot(sessions, npcs, out / cfg["analysis"]["figure"], cfg["analysis"]["smooth_window"])
    print(text)
    print(f"\nsaved {out / 'human_summary.csv'}, human_curve.csv, human_summary.txt, {fig}")


if __name__ == "__main__":
    main()

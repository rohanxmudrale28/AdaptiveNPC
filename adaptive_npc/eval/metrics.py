"""Match metrics and across-seed aggregation (mean +- 95% t-interval)."""
import numpy as np
from scipy import stats


def summarize_rounds(rounds):
    """rounds: dicts with winner (0 NPC / 1 player / -1 draw), steps, ret_npc."""
    w = np.array([r["winner"] for r in rounds])
    return {"win_rate": float(np.mean(np.where(w == 0, 1.0, np.where(w == -1, 0.5, 0.0)))),
            "wins": float(np.mean(w == 0)), "draws": float(np.mean(w == -1)), "losses": float(np.mean(w == 1)),
            "mean_return": float(np.mean([r["ret_npc"] for r in rounds])),
            "mean_steps": float(np.mean([r["steps"] for r in rounds])), "rounds": len(rounds)}


def mean_ci(values, conf=0.95):
    """(mean, half-width of the t-interval); half-width is nan for fewer than 2 values."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    sem = stats.sem(v)
    return float(v.mean()), float(sem * stats.t.ppf((1 + conf) / 2, len(v) - 1))


def aggregate(rows, metrics=("win_rate", "mean_return"), keys=("method", "opponent")):
    """Group per-seed rows by `keys`; one output row per group with <metric>_mean / <metric>_ci95."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        row = dict(zip(keys, g))
        row["n_seeds"] = len(rs)
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_ci95"] = mean_ci([r[m] for r in rs])
        out.append(row)
    return out

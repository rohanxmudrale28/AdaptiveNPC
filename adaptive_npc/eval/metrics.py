"""Match metrics and across-seed aggregation (mean +- 95% t-interval)."""
import numpy as np
from scipy import stats


def _win_score(w):
    return np.where(w == 0, 1.0, np.where(w == -1, 0.5, 0.0))


def summarize_rounds(rounds, hp_max, first_n):
    """rounds: dicts with winner (0 NPC / 1 player / -1 draw), steps, ret_npc, hp_npc.
    mean_steps = episode length; damage_taken = NPC HP lost per round; win_rate_first_n = win rate
    over the first `first_n` rounds of the match."""
    w = np.array([r["winner"] for r in rounds])
    return {"win_rate": float(np.mean(_win_score(w))),
            "win_rate_first_n": float(np.mean(_win_score(w[:first_n]))),
            "damage_taken": float(np.mean([hp_max - r["hp_npc"] for r in rounds])),
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


def aggregate(rows, metrics=("win_rate", "win_rate_first_n", "mean_return", "mean_steps", "damage_taken"), keys=("method", "opponent")):
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

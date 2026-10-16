"""Round-robin balance matrix and its checks. Pure functions; scripts/balance_matrix.py drives them.

Matrices are indexed [row, col]. Win rates count draws as 1/2.
  slot0[i, j]   = score of i when i plays the NPC slot (player 0) against j (player 1)
  winrate[i, j] = score of i against j, averaged over both side assignments
  decisive[i,j] = wins / (wins + losses) of i vs j, both sides (0.5 if every round was a draw)
  draw[i, j]    = draw rate of the unordered pair, pooled over both side assignments
Dominance / counter checks look at BOTH winrate and decisive and take the conservative reading.
"""
import numpy as np

from adaptive_npc.env.match import run_match
from adaptive_npc.players.base import RandomPlayer
from adaptive_npc.players.scripted import make_player


def make_any(name, players_cfg, env_cfg, seed):
    if name == "random":
        return RandomPlayer(seed)
    return make_player(name, players_cfg, env_cfg, seed)


def play_pair(task):
    """task = (env_cfg, players_cfg, name0, name1, seed, n_rounds) -> (wins0, wins1, draws)."""
    env_cfg, players_cfg, n0, n1, seed, n_rounds = task
    p0 = make_any(n0, players_cfg, env_cfg, seed + 1000)
    p1 = make_any(n1, players_cfg, env_cfg, seed + 2000)
    res = run_match(env_cfg, p0, p1, n_rounds=n_rounds, seed=seed)
    w0 = sum(r["winner"] == 0 for r in res)
    w1 = sum(r["winner"] == 1 for r in res)
    return w0, w1, len(res) - w0 - w1


def build_matrices(counts):
    """counts[i, j] = (wins of i as slot 0, wins of j as slot 1, draws), pooled over seeds."""
    w0, w1, d = counts[..., 0], counts[..., 1], counts[..., 2]
    n = w0 + w1 + d
    slot0 = (w0 + 0.5 * d) / n
    winrate = (slot0 + (1.0 - slot0.T)) / 2.0
    draw = (d + d.T) / (n + n.T)
    wins = w0 + w1.T                      # wins of i vs j over both side assignments
    losses = w1 + w0.T
    decisive = np.where(wins + losses > 0, wins / np.maximum(wins + losses, 1), 0.5)
    return slot0, winrate, draw, decisive


def run_checks(names, slot0, winrate, draw, decisive, cfg):
    """Return ([(check_name, passed, detail)], flagged_draw_pairs[(a, b, rate)])."""
    n = len(names)
    exempt = set(cfg.get("non_random_exclude", []))
    active = [i for i in range(n) if names[i] not in exempt]
    lo, hi = cfg["mirror_tolerance"]
    margin = cfg.get("win_margin", 0.0)
    checks = []

    bad = [(names[i], slot0[i, i]) for i in range(n) if not lo <= slot0[i, i] <= hi]
    checks.append((f"mirror matches in [{lo:.2f}, {hi:.2f}]", not bad,
                   "; ".join(f"{a}={v:.3f}" for a, v in bad) or "all ok"))

    thr = cfg["dominance_threshold"]
    dom = []
    for i in active:
        others = [j for j in range(n) if j != i]
        by_adj = all(winrate[i, j] >= thr for j in others)
        by_dec = all(decisive[i, j] >= thr for j in others)
        if by_adj or by_dec:
            dom.append(f"{names[i]} (min adj {min(winrate[i, j] for j in others):.3f}, "
                       f"min decisive {min(decisive[i, j] for j in others):.3f})")
    checks.append((f"no dominant archetype (>= {thr:.2f} vs all others, adjusted or decisive)",
                   not dom, ", ".join(dom) or "none dominant"))

    no_win = [names[i] for i in active
              if not any(winrate[i, j] > 0.5 + margin for j in range(n) if j != i)]
    no_loss = [names[i] for i in active
               if not any(winrate[i, j] < 0.5 - margin for j in range(n) if j != i)]
    checks.append(("every non-random archetype wins >=1 and loses >=1 matchup",
                   not no_win and not no_loss,
                   f"no win: {no_win or '-'}; no loss: {no_loss or '-'}"))

    uncountered, best = [], []
    for i in active:
        # a counter must reach the threshold on BOTH metrics
        w, who = max((min(winrate[j, i], decisive[j, i]), names[j]) for j in active if j != i)
        best.append(f"{names[i]}<-{who} adj {winrate[names.index(who), i]:.2f}/dec {decisive[names.index(who), i]:.2f}")
        if w < cfg["counter_threshold"]:
            uncountered.append(f"{names[i]} (best counter {who} adj {winrate[names.index(who), i]:.3f}, "
                               f"decisive {decisive[names.index(who), i]:.3f})")
    checks.append((f"every non-random archetype has a counter >= {cfg['counter_threshold']:.2f}",
                   not uncountered, "; ".join(uncountered) or "; ".join(best)))

    flagged = [(names[i], names[j], float(draw[i, j])) for i in range(n) for j in range(i, n)
               if draw[i, j] > cfg["draw_flag_threshold"]]
    return checks, flagged

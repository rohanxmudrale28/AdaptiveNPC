"""Random-action augmentation for Γ's training set.

Start states come ONLY from the training rollouts (never val / held-out data), both sides' actions are uniformly
random, and the labels are exact (true simulator). Part of the starts yield two chained transitions (the second one
starts from the true state reached by the first), so states reached by random play are covered too.
Start states in collision-heavy places (players within Manhattan distance 2, or anyone on the arena edge) are
sampled `hard_weight` times as often.
"""
import numpy as np

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.models.env_model import X0, X1, Y0, Y1, TrueSimulator

KEYS = ("state", "next_state", "a_npc", "a_opp", "reward")


def hard_mask(size, s):
    near = (np.abs(s[:, X0] - s[:, X1]) + np.abs(s[:, Y0] - s[:, Y1])) <= 2
    edge = np.zeros(len(s), bool)
    for c in (X0, Y0, X1, Y1):
        edge |= (s[:, c] == 0) | (s[:, c] == size - 1)
    return near | edge


def augment_random_actions(env_cfg, pool, n_base, fraction, two_step_share, hard_weight, seed):
    """Transitions amounting to `fraction` of (base + augmentation); `n_base` is the on-policy training size.
    Returns a dict with KEYS arrays."""
    rng = np.random.default_rng(seed)
    sim = TrueSimulator(env_cfg)
    n_total = int(round(n_base * fraction / (1.0 - fraction)))
    n_two = int(n_total * two_step_share) // 2  # each yields up to 2 transitions
    n_one = n_total - 2 * n_two
    w = 1.0 + (hard_weight - 1.0) * hard_mask(int(env_cfg["arena_size"]), pool)
    starts = pool[rng.choice(len(pool), n_one + n_two, p=w / w.sum())]
    a = rng.integers(N_ACTIONS, size=(4, len(starts)))
    s1, r1, d1 = sim.step(starts, a[0], a[1])
    k = np.arange(n_one, len(starts))  # starts that continue to a second step
    k = k[~d1[k]]
    s2, r2, _ = sim.step(s1[k], a[2][k], a[3][k])
    return {"state": np.concatenate([starts, s1[k]]), "next_state": np.concatenate([s1, s2]),
            "a_npc": np.concatenate([a[0], a[2][k]]), "a_opp": np.concatenate([a[1], a[3][k]]),
            "reward": np.concatenate([r1, r2]).astype(np.float32)}


def mix(base, aug):
    """Concatenate on-policy training data with augmentation (only the keys training needs)."""
    return {k: np.concatenate([base[k], aug[k].astype(base[k].dtype)]) for k in KEYS}

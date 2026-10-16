import numpy as np

from adaptive_npc.eval.balance import build_matrices, run_checks

CFG = {"non_random_exclude": ["random"], "dominance_threshold": 0.65,
       "mirror_tolerance": [0.45, 0.55], "draw_flag_threshold": 0.5,
       "counter_threshold": 0.6, "win_margin": 0.0}


def _counts_from_slot0(slot0, draws=None, n=100):
    k = len(slot0)
    d = np.zeros((k, k)) if draws is None else draws * n
    w0 = slot0 * n - 0.5 * d
    w1 = n - w0 - d
    return np.stack([w0, w1, d], axis=-1)


def test_rock_paper_scissors_passes():
    # a beats b, b beats c, c beats a at 80%; random is beaten by everyone
    names = ["a", "b", "c", "random"]
    s = np.full((4, 4), 0.5)
    for x, y in [(0, 1), (1, 2), (2, 0)]:
        s[x, y], s[y, x] = 0.8, 0.2
    s[:3, 3], s[3, :3] = 0.9, 0.1
    slot0, wr, draw, dec = build_matrices(_counts_from_slot0(s))
    checks, flagged = run_checks(names, slot0, wr, draw, dec, CFG)
    assert all(ok for _, ok, _ in checks), checks
    assert flagged == []


def test_dominant_and_uncountered_archetype_fails():
    names = ["a", "b", "random"]
    s = np.full((3, 3), 0.5)
    s[0, 1], s[1, 0] = 0.9, 0.1
    s[0, 2], s[2, 0] = 0.9, 0.1
    s[1, 2], s[2, 1] = 0.7, 0.3
    slot0, wr, draw, dec = build_matrices(_counts_from_slot0(s))
    res = {name: ok for name, ok, _ in run_checks(names, slot0, wr, draw, dec, CFG)[0]}
    assert not res["no dominant archetype (>= 0.65 vs all others, adjusted or decisive)"]
    assert not res["every non-random archetype wins >=1 and loses >=1 matchup"]
    assert not res["every non-random archetype has a counter >= 0.60"]


def test_draw_flag_and_mirror_bias():
    names = ["a", "b"]
    s = np.array([[0.7, 0.5], [0.5, 0.5]])  # a has side bias in the mirror
    draws = np.array([[0.0, 0.6], [0.6, 0.0]])
    slot0, wr, draw, dec = build_matrices(_counts_from_slot0(s, draws))
    checks, flagged = run_checks(names, slot0, wr, draw, dec, CFG)
    assert not checks[0][1]
    assert [(a, b) for a, b, _ in flagged] == [("a", "b")]


def test_decisive_rate_ignores_draws():
    # a vs b: 10 wins, 5 losses, 85 draws (a as slot 0 only matters for the pooled counts)
    counts = np.zeros((2, 2, 3))
    counts[0, 1] = (10, 5, 85)
    counts[1, 0] = (5, 10, 85)   # mirrored side assignment
    counts[0, 0] = counts[1, 1] = (1, 1, 98)
    slot0, wr, draw, dec = build_matrices(counts)
    assert abs(dec[0, 1] - 20 / 30) < 1e-9
    assert wr[0, 1] < 0.6 < dec[0, 1]

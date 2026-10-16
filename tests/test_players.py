from itertools import combinations

import numpy as np
import pytest

from adaptive_npc.config import load_config
from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.env.match import MatchHook, run_match
from adaptive_npc.npc.scripted_npc import make_scripted_npc
from adaptive_npc.players.base import RandomPlayer
from adaptive_npc.players.scripted import make_player

ARCHETYPES = ["rusher", "camper", "repeater", "mixer", "switcher", "flanker", "kiter", "counter_puncher"]


@pytest.fixture(scope="module")
def env_cfg():
    return load_config("configs/env.yaml")


@pytest.fixture(scope="module")
def pcfg():
    return load_config("configs/players.yaml")


class Recorder(MatchHook):
    def __init__(self):
        self.h = [np.zeros(N_ACTIONS), np.zeros(N_ACTIONS)]

    def on_step(self, round_idx, game, a0, a1):
        self.h[0][a0] += 1
        self.h[1][a1] += 1


def histogram(env_cfg, pcfg, name, side=1, rounds=40, seed=0):
    """Normalised action histogram of archetype `name` playing `side` against a random player."""
    arch, rnd = make_player(name, pcfg, env_cfg, seed), RandomPlayer()
    rec = Recorder()
    if side == 1:
        run_match(env_cfg, rnd, arch, n_rounds=rounds, seed=seed, hook=rec)
    else:
        run_match(env_cfg, arch, rnd, n_rounds=rounds, seed=seed, hook=rec)
    return rec.h[side] / rec.h[side].sum()


def test_split_matches_config(pcfg):
    assert set(pcfg["train"]) | set(pcfg["heldout"]) == set(ARCHETYPES)
    assert not set(pcfg["train"]) & set(pcfg["heldout"])


def test_action_histograms_differ(env_cfg, pcfg):
    hists = {n: histogram(env_cfg, pcfg, n) for n in ARCHETYPES}
    for a, b in combinations(ARCHETYPES, 2):
        l1 = np.abs(hists[a] - hists[b]).sum()
        assert l1 > 0.15, f"{a} vs {b}: L1 distance {l1:.3f}"


@pytest.mark.parametrize("name", ["rusher", "camper", "repeater", "mixer", "kiter", "counter_puncher"])
def test_archetype_plays_either_side(env_cfg, pcfg, name):
    h1 = histogram(env_cfg, pcfg, name, side=1, rounds=60)
    h0 = histogram(env_cfg, pcfg, name, side=0, rounds=60)
    assert np.abs(h0 - h1).sum() < 0.3


def test_switcher_switches_on_schedule(env_cfg, pcfg):
    sw = make_player("switcher", pcfg, env_cfg, 0)
    every, cycle = pcfg["switcher"]["switch_every_rounds"], pcfg["switcher"]["cycle"]
    names = []
    for _ in range(3 * every + 5):
        sw.reset()
        names.append(sw.current_name)
    for r, n in enumerate(names):
        assert n == cycle[(r // every) % len(cycle)]
    assert names[0] != names[every] and names[every - 1] == names[0]


def test_switcher_within_round_switch(env_cfg, pcfg):
    cfg = dict(pcfg, switcher=dict(pcfg["switcher"], switch_at_step=75))
    sw = make_player("switcher", cfg, env_cfg, 0)
    sw.reset()
    obs = np.zeros(21, dtype=np.float32)
    obs[2], obs[4], obs[8] = 1 / 8, 1.0, 1.0
    used = []
    for t in (0, 74, 75, 149):
        used.append(sw._index(t))
    assert used[0] == used[1] != used[2] == used[3]
    assert sw.act(obs, [(0, 0)] * 80) in range(N_ACTIONS)


def test_scripted_npc_beats_random(env_cfg, pcfg):
    npc = make_scripted_npc(pcfg, env_cfg, 0)
    res = run_match(env_cfg, npc, RandomPlayer(), n_rounds=200, seed=1)
    wins = np.mean([r["winner"] == 0 for r in res])
    assert wins > 0.8, wins
    # and as player 1 (mirrored observations)
    res = run_match(env_cfg, RandomPlayer(), make_scripted_npc(pcfg, env_cfg, 0), n_rounds=200, seed=2)
    assert np.mean([r["winner"] == 1 for r in res]) > 0.8


def test_players_are_seeded(env_cfg, pcfg):
    for n in ARCHETYPES:
        a = histogram(env_cfg, pcfg, n, seed=3)
        b = histogram(env_cfg, pcfg, n, seed=3)
        assert np.array_equal(a, b), n

"""M8: MBOM components (no checkpoints needed unless marked)."""
import numpy as np
import pytest
import torch

from adaptive_npc.config import load_config
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.observations import encode
from adaptive_npc.models.env_model import STATE_DIM, EnvSpec, raw_state
from adaptive_npc.npc.mbom.mixer import BayesMixer
from adaptive_npc.npc.mbom.state_obs import encode_raw

ENV = load_config("configs/env.yaml")
SPEC = EnvSpec(ENV)


def random_legal_states(n, seed):
    """Uniform over each channel's legal range (positions may coincide; encode does not care)."""
    rng = np.random.default_rng(seed)
    return rng.integers(SPEC.lo, SPEC.hi.astype(np.int64) + 1, size=(n, STATE_DIM)).astype(np.int16)


def set_game(g, s):
    g.x, g.y = [int(s[0]), int(s[2])], [int(s[1]), int(s[3])]
    g.face, g.hp = [int(s[4]), int(s[5])], [int(s[6]), int(s[7])]
    g.sta, g.cd, g.stun = [int(s[8]), int(s[9])], [int(s[10]), int(s[11])], [int(s[12]), int(s[13])]
    g.t = int(s[14])


def test_encode_raw_matches_encode_both_views():
    raw = random_legal_states(1000, 0)
    g = ArenaGame(ENV, rng=np.random.default_rng(0))
    for who in (0, 1):
        ref = []
        for s in raw:
            set_game(g, s)
            assert np.array_equal(raw_state(g), s)
            ref.append(encode(g, who))
        got = encode_raw(raw, who, SPEC).numpy()
        assert got.dtype == np.float32
        assert np.array_equal(got, np.stack(ref))


def test_encode_raw_on_real_rollout_states():
    g = ArenaGame(ENV, rng=np.random.default_rng(3))
    rng = np.random.default_rng(4)
    states, ref = [], []
    for _ in range(5):
        g.reset()
        done = False
        while not done:
            states.append(raw_state(g))
            ref.append(encode(g, 0))
            _, done = g.step(*(int(a) for a in rng.integers(8, size=2)))
    got = encode_raw(torch.as_tensor(np.stack(states)), 0, ENV)  # accepts a tensor and a raw env cfg
    assert np.array_equal(got.numpy(), np.stack(ref))


# ---- mixer (synthetic: 3 fixed categorical "IOPs", actions sampled from one generating level) ----

def synthetic_levels(peak=0.4):
    """Level m puts `peak` on action 2m and spreads the rest uniformly: overlapping but distinct."""
    d = np.full((3, 8), (1 - peak) / 7)
    for m in range(3):
        d[m, 2 * m] = peak
    return d


def run_mixer(mixer, levels, gen_seq, seed=0):
    rng = np.random.default_rng(seed)
    alphas = []
    for g in gen_seq:
        a = rng.choice(8, p=levels[g])
        mixer.update(levels[:, a])
        alphas.append(mixer.alpha)
    return np.array(alphas)


def stable_from(argmaxes, level):
    """First index from which `level` leads for good (len if it never does)."""
    off = np.nonzero(argmaxes != level)[0]
    return int(off.max()) + 1 if len(off) else 0


@pytest.mark.parametrize("peak,max_median_lag,max_lag", [(0.4, 60, 120), (0.6, 30, 40)])
def test_mixer_converges_to_generating_level_and_follows_a_switch(peak, max_median_lag, max_lag):
    """Default mixer settings, 20 seeds. The switch lag (steps until the new level leads for good) is dominated by
    the sticky prior (moving average of posteriors, floored at prior_floor), not by the evidence: measured median /
    max 49.5 / 98 steps for peak 0.4 and 20 / 26 for peak 0.6. Initial convergence: stable lead by step 17."""
    levels, lags = synthetic_levels(peak), []
    for seed in range(20):
        mixer = BayesMixer(3, decay=0.9, temperature=0.5, prior_avg=0.9, eps=1e-3, prior_floor=0.01)
        al = run_mixer(mixer, levels, [1] * 300 + [2] * 300, seed)
        assert stable_from(al[:300].argmax(1), 1) <= 50, f"seed {seed}: level 1 not leading for good by step 50"
        # steady state sits a little under the cap (prior floor + overlapping likelihoods: measured 0.71-0.77)
        assert al[200:300, 1].mean() > 0.85 * mixer.alpha_cap
        lag = stable_from(al[300:].argmax(1), 2)
        assert lag < 300, f"seed {seed}: never switched"
        assert al[500:, 2].mean() > 0.85 * mixer.alpha_cap
        lags.append(lag)
    assert np.median(lags) <= max_median_lag and max(lags) <= max_lag, f"switch lags {sorted(lags)}"


def test_mixer_alpha_cap():
    mixer = BayesMixer(3, temperature=0.5)
    assert mixer.alpha_cap == pytest.approx(np.e ** 2 / (np.e ** 2 + 2))  # ≈ 0.787
    assert 0.78 < mixer.alpha_cap < 0.79
    al = run_mixer(mixer, synthetic_levels(peak=0.9), [0] * 500)
    assert al.max() <= mixer.alpha_cap + 1e-12
    assert al[-1, 0] == pytest.approx(mixer.alpha_cap, abs=1e-3)
    assert np.allclose(al.sum(1), 1.0)


def test_mixer_mode_flags():
    levels = synthetic_levels()
    uni = run_mixer(BayesMixer(3, mode="uniform"), levels, [1] * 100)
    assert np.allclose(uni, 1 / 3)
    for m in range(3):
        fixed = run_mixer(BayesMixer(3, mode="fixed", fixed_level=m), levels, [(m + 1) % 3] * 100)
        assert np.all(fixed == np.eye(3)[m])
    mixer = BayesMixer(3, mode="fixed", fixed_level=2)
    assert np.allclose(mixer.mix(levels), levels[2])
    with pytest.raises(ValueError):
        BayesMixer(3, mode="fixed")
    with pytest.raises(ValueError):
        BayesMixer(3, mode="softmax")


def test_mixer_mix_is_alpha_weighted_and_reset_restores_uniform():
    levels = synthetic_levels()
    mixer = BayesMixer(3)
    run_mixer(mixer, levels, [0] * 50)
    assert np.allclose(mixer.mix(levels), mixer.alpha @ levels)
    assert mixer.mix(levels).sum() == pytest.approx(1.0)
    mixer.reset()
    assert np.allclose(mixer.alpha, 1 / 3)

"""M8: MBOM components (no checkpoints needed unless marked)."""
import numpy as np
import torch

from adaptive_npc.config import load_config
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.observations import encode
from adaptive_npc.models.env_model import STATE_DIM, EnvSpec, raw_state
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

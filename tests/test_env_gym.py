import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from adaptive_npc.config import load_config
from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.env.match import npc_win_rate, run_match
from adaptive_npc.env.vec_env import make_vec_env
from adaptive_npc.players.base import RandomPlayer


@pytest.fixture
def cfg():
    return load_config("configs/env.yaml")


def _rollout(cfg, seed, n=300):
    env = NPCEnv(cfg, seed=seed)
    obs, _ = env.reset()
    acts = np.random.default_rng(99).integers(8, size=n)
    traj = [obs]
    for a in acts:
        obs, r, done, trunc, info = env.step(int(a))
        traj.append((obs, r, done))
        if done:
            obs, _ = env.reset()
            traj.append(obs)
    return traj


def _flat(traj):
    return np.concatenate([np.ravel(np.asarray(x, dtype=np.float64)) if not isinstance(x, tuple)
                           else np.concatenate([x[0], [x[1], x[2]]]) for x in traj])


def test_same_seed_same_trajectory(cfg):
    assert np.array_equal(_flat(_rollout(cfg, 7)), _flat(_rollout(cfg, 7)))
    assert not np.array_equal(_flat(_rollout(cfg, 7)), _flat(_rollout(cfg, 8)))


def test_check_env(cfg):
    check_env(NPCEnv(cfg, seed=0))
    sb3_check_env(NPCEnv(cfg, seed=0))


def _vec_run(cfg, vec_type, n_envs=4, steps=60):
    venv = make_vec_env(cfg, n_envs=n_envs, vec_env_type=vec_type, seed=3)
    obs = [venv.reset()]
    acts = np.random.default_rng(0).integers(8, size=(steps, n_envs))
    for a in acts:
        o, r, d, _ = venv.step(a)
        obs.append(np.concatenate([o, r[:, None], d[:, None]], axis=1))
    venv.close()
    return np.stack(obs[1:]), obs[0]


def test_vec_env_reproducible_and_envs_differ(cfg):
    a, a0 = _vec_run(cfg, "dummy")
    b, b0 = _vec_run(cfg, "dummy")
    assert np.array_equal(a, b) and np.array_equal(a0, b0)
    # sub-envs are seeded base_seed + i, so they must not all share a trajectory
    assert not np.array_equal(a[:, 0], a[:, 1])
    assert len({tuple(row) for row in a0.tolist()}) == 4  # distinct initial states


def test_vec_env_matches_single_env_seeding(cfg):
    venv = make_vec_env(cfg, n_envs=3, vec_env_type="dummy", seed=10)
    first = venv.reset()
    for i in range(3):
        o, _ = NPCEnv(cfg, seed=10 + i).reset()
        assert np.array_equal(first[i], o)


def test_subproc_matches_dummy(cfg):
    d, _ = _vec_run(cfg, "dummy", n_envs=2, steps=30)
    s, _ = _vec_run(cfg, "subproc", n_envs=2, steps=30)
    assert np.array_equal(d, s)


def test_match_runs_and_is_reproducible(cfg):
    res1 = run_match(cfg, RandomPlayer(), RandomPlayer(), n_rounds=5, seed=4)
    res2 = run_match(cfg, RandomPlayer(), RandomPlayer(), n_rounds=5, seed=4)
    assert res1 == res2 and len(res1) == 5
    assert all(r["winner"] in (0, 1, -1) and 1 <= r["steps"] <= cfg["max_steps"] for r in res1)
    assert 0.0 <= npc_win_rate(res1) <= 1.0

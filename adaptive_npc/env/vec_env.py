"""Seeded vectorized envs (SB3 DummyVecEnv / SubprocVecEnv) built from config."""
from functools import partial

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.seeding import env_seeds


def _make_env(cfg, seed, opponent_factory, wrapper_factory):
    opponent = opponent_factory() if opponent_factory is not None else None
    env = NPCEnv(cfg, opponent=opponent, seed=seed)
    return wrapper_factory(env) if wrapper_factory is not None else env


def make_vec_env(cfg, n_envs=None, vec_env_type=None, seed=None, opponent_factory=None,
                 wrapper_factory=None):
    """Build a vec env whose i-th sub-env is seeded with base_seed + i.

    Defaults come from cfg['vec'] and cfg['seed']. `opponent_factory` is a picklable
    zero-arg callable returning a fresh Player per sub-env (default: RandomPlayer).
    `wrapper_factory` (picklable, env -> env) wraps each sub-env, e.g. for prediction conditioning.
    """
    vec = cfg.get("vec", {})
    n_envs = int(vec.get("n_envs", 1) if n_envs is None else n_envs)
    vec_env_type = vec.get("vec_env_type", "dummy") if vec_env_type is None else vec_env_type
    base_seed = int(cfg.get("seed", 0) if seed is None else seed)
    fns = [partial(_make_env, cfg, s, opponent_factory, wrapper_factory) for s in env_seeds(base_seed, n_envs)]
    if vec_env_type == "dummy":
        return DummyVecEnv(fns)
    if vec_env_type == "subproc":
        return SubprocVecEnv(fns, start_method="spawn")
    raise ValueError(f"vec_env_type must be 'dummy' or 'subproc', got {vec_env_type!r}")

"""Seed python, numpy, torch (and provide a seeded Generator for gymnasium envs)."""
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed all global RNGs. Returns a numpy Generator seeded the same way."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def env_seeds(base_seed, n_envs):
    """Distinct, reproducible per-env seeds: base_seed + i."""
    return [int(base_seed) + i for i in range(n_envs)]

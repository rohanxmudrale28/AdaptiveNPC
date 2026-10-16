import random

import numpy as np
import torch

from adaptive_npc.seeding import env_seeds, seed_everything


def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seed_everything_reproducible():
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    assert a == _draw()
    seed_everything(124)
    assert a != _draw()


def test_env_seeds_distinct():
    assert env_seeds(10, 4) == [10, 11, 12, 13]

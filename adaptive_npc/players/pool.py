"""Opponent pool: each round samples one archetype from a list (used to train the PPO NPC)."""
import numpy as np

from adaptive_npc.players.base import Player
from adaptive_npc.players.scripted import make_player


class PoolPlayer(Player):
    def __init__(self, names, players_cfg, env_cfg, seed=None):
        self.names = list(names)
        base = players_cfg.get("seed", 0) if seed is None else seed
        self.members = [make_player(n, players_cfg, env_cfg, base + 1 + i) for i, n in enumerate(self.names)]
        self.rng = np.random.default_rng(base)
        self._cur = 0

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        for i, m in enumerate(self.members):
            m.seed(seed + 1 + i)

    def reset(self):
        for m in self.members:  # all reset so stateful members (switcher) keep their round counters
            m.reset()
        self._cur = int(self.rng.integers(len(self.members)))

    @property
    def current_name(self):
        return self.names[self._cur]

    def act(self, obs, hist):
        return self.members[self._cur].act(obs, hist)


class PoolFactory:
    """Picklable zero-arg factory for make_vec_env(opponent_factory=...)."""

    def __init__(self, names, players_cfg, env_cfg):
        self.args = (list(names), players_cfg, env_cfg)

    def __call__(self):
        return PoolPlayer(*self.args)

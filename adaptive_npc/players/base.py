"""Player interface shared by scripted players, learners, humans, and the NPC itself.

A player always sees the game from its own point of view (observations.encode(game, who)).
`hist` is the list of (own_action, other_action) tuples for the current round.
"""
from typing import NamedTuple

import numpy as np

from adaptive_npc.env.actions import N_ACTIONS


class Transition(NamedTuple):
    obs: np.ndarray
    action: int
    opp_action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class Player:
    learns = False  # True -> env/match encode the next obs and call observe() every step

    def seed(self, seed):
        """Re-seed any internal randomness."""

    def reset(self):
        """Called at the start of every round. Learned state must persist across rounds."""

    def act(self, obs, hist):
        raise NotImplementedError

    def observe(self, transition):
        """Called after every step if `learns` is True."""


class RandomPlayer(Player):
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, hist):
        return int(self.rng.integers(N_ACTIONS))

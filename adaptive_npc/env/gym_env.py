"""Gymnasium wrapper: the agent controls the NPC (player 0); any Player is embedded as the opponent."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.observations import OBS_DIM, encode
from adaptive_npc.players.base import RandomPlayer, Transition


class NPCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, opponent=None, seed=None):
        super().__init__()
        self.cfg = cfg
        self.game = ArenaGame(cfg)
        self.opponent = opponent if opponent is not None else RandomPlayer()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._init_seed = int(cfg.get("seed", 0)) if seed is None else int(seed)
        self._hist = []  # opponent-view history: (opp_action, npc_action)
        self._opp_obs = None
        self._pending_opp = None

    def peek_opp_action(self):
        """The opponent's action for the coming step (computed once, then used by step()).
        Lets oracle-conditioned NPCs see it; the opponent's RNG order is unchanged."""
        if self._pending_opp is None:
            self._pending_opp = int(self.opponent.act(self._opp_obs, self._hist))
        return self._pending_opp

    def reset(self, *, seed=None, options=None):
        if seed is None and self._init_seed is not None:
            seed, self._init_seed = self._init_seed, None
        super().reset(seed=seed)
        if seed is not None:
            self.opponent.seed(int(self.np_random.integers(2**31 - 1)))
        self.game.rng = self.np_random
        self.game.reset()
        self.opponent.reset()
        self._hist = []
        self._pending_opp = None
        self._opp_obs = encode(self.game, 1)
        return encode(self.game, 0), {}

    def step(self, action):
        g = self.game
        a1 = self.peek_opp_action()
        self._pending_opp = None
        a0 = int(action)
        reward, done = g.step(a0, a1)
        self._hist.append((a1, a0))
        obs = encode(g, 0)
        next_opp_obs = encode(g, 1)
        if self.opponent.learns:
            self.opponent.observe(Transition(self._opp_obs, a1, a0, -reward, next_opp_obs, done))
        self._opp_obs = next_opp_obs
        info = {"opp_action": a1}
        if done:
            info["winner"] = g.winner
            info["hp"] = (g.hp[0], g.hp[1])
        return obs, reward, done, False, info

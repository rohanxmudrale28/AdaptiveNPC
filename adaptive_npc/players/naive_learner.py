"""Naive learner: a pretrained PPO policy that keeps updating online (ppo_lite) during a match.

Acts stochastically (PPO is on-policy). Every `update_every` steps the buffer is consumed; if the round
is still running, the update waits for the next act() so the bootstrap value V(next input) is exact.
With learn=False it is the frozen control (same policy, no updates).
"""
import copy

import numpy as np

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.models.ppo_lite import PPOLite, action_probs, probs_and_value
from adaptive_npc.npc.ppo_npc import load_policy
from adaptive_npc.players.base import Player


class NaiveLearner(Player):
    learns = True  # observe() is needed even when frozen (the reasoning learner tracks actions there)

    def __init__(self, policy_path, ppo_cfg, update_every, learn=True, seed=0):
        self.policy = load_policy(policy_path)
        self.learn = bool(learn)
        self.update_every = int(update_every)
        self.ppo = PPOLite(self.policy, ppo_cfg, seed) if self.learn else None
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.update_stats = []
        self.record_inputs = None  # set to a list to record (obs, extra) at every act() (probe states)
        self._pending = None
        self._update_due = False

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def _input(self, obs):
        """Policy input for this step (the reasoning learner appends its NPC prediction)."""
        return np.asarray(obs, dtype=np.float32)

    def _probe_extra(self):
        return None

    def act(self, obs, hist):
        x = self._input(obs)
        probs, value = probs_and_value(self.policy, x)
        if self._update_due:  # buffer full mid-round: bootstrap with V(x) under the pre-update policy
            self._update_due = False
            self.update_stats.append(self.ppo.update(value))
            probs, value = probs_and_value(self.policy, x)
        if self.record_inputs is not None:
            self.record_inputs.append((np.asarray(obs, dtype=np.float32), self._probe_extra()))
        a = int(self.rng.choice(N_ACTIONS, p=probs))
        self._pending = (x, a, float(np.log(max(probs[a], 1e-12))), value)
        return a

    def observe(self, transition):
        if not self.learn or self._pending is None:
            return
        x, a, logp, value = self._pending
        self._pending = None
        self.ppo.record(x, a, logp, value, transition.reward, transition.done)
        self.steps += 1
        if self.steps % self.update_every == 0:
            if transition.done:
                self.update_stats.append(self.ppo.update(0.0))
            else:
                self._update_due = True

    @property
    def n_updates(self):
        return self.ppo.n_updates if self.ppo else 0

    def policy_probs(self, inputs, policy=None):
        """Action probabilities of `policy` (default: current) on a batch of policy inputs."""
        return action_probs(policy or self.policy, inputs)

    def frozen_copy(self):
        """Deep copy that no longer updates (for end-of-match evaluation)."""
        twin = copy.deepcopy(self)
        twin.learn, twin.ppo, twin._update_due, twin._pending = False, None, False, None
        twin.record_inputs = None
        return twin

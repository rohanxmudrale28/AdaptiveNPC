"""Reasoning learner: keeps its own GRU model of the NPC and conditions its policy on the prediction.

Policy input = [player-view obs, p̂(NPC next action)]. The NPC model is a PlayerPredictor trained on
scripted-NPC actions (configs/learners_npc_model.yaml); from the player side the "owner" is the learner and
the "other" is the NPC, so OnlinePredictor.observe(own_action, npc_action). Online, the GRU finetunes
with `gru_online` (predictor.yaml 'online' block) and the policy with ppo_lite. learn=False freezes both.
"""
import numpy as np

from adaptive_npc.models.player_model import OnlinePredictor, load_predictor
from adaptive_npc.players.naive_learner import NaiveLearner


class ReasoningLearner(NaiveLearner):
    def __init__(self, policy_path, npc_model_path, ppo_cfg, update_every, gru_online=None, learn=True, seed=0):
        super().__init__(policy_path, ppo_cfg, update_every, learn, seed)
        model, model_cfg = load_predictor(npc_model_path)
        self.predictor = OnlinePredictor(model, int(model_cfg["window"]), gru_online if learn else None)
        self.pred_hits = []  # 1 if argmax prediction matched the NPC's action, per step
        self._pred = None

    def reset(self):
        self.predictor.reset_round()

    def _input(self, obs):
        self._pred = self.predictor.predict(obs).astype(np.float32)
        return np.concatenate([np.asarray(obs, dtype=np.float32), self._pred])

    def _probe_extra(self):
        return self.predictor._pending.clone()  # GRU input window of this step

    def observe(self, transition):
        if self._pred is not None:
            self.pred_hits.append(int(np.argmax(self._pred) == transition.opp_action))
            self._pred = None
        self.predictor.observe(transition.action, transition.opp_action)
        super().observe(transition)

    def frozen_copy(self):
        twin = super().frozen_copy()
        twin.predictor.ft, twin.predictor.opt, twin.predictor.buffer = None, None, None
        return twin

"""Lookahead opponent: the opponent MBOM's imagination is built to model (test H2, PLAN.md M8).

Each step it predicts the NPC's action with the frozen NPC policy (the cell's ppo_predicted checkpoint) fed by the
pretrained, frozen φ0, i.e. it assumes the NPC is plain ppo_predicted. It then picks the action that maximizes the
imagined return of eq. 5 from its own side: for each of its 8 candidate actions, k-step continuations with uniformly
random own actions while the NPC acts with π∘φ0, return Σ γ^t rᵒ_t + γ^{k+1}·(-V), max over continuations. This is
`imagined_returns` (npc/mbom/imagination.py) with the same k / samples / bootstrap as MBOM's imagination, run in the
true simulator (or Γ). With probability ε it plays a uniformly random action instead. No online learning.
It reconstructs the raw state exactly from its own (player-view) observation.
"""
import numpy as np
import torch
import torch.nn.functional as F

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.models.env_model import EnvSpec, load_env_model
from adaptive_npc.models.player_model import OnlinePredictor, load_predictor
from adaptive_npc.npc.mbom.imagination import aggregate_q, imagined_returns, policy_fns
from adaptive_npc.npc.mbom.state_obs import decode_obs, encode_raw
from adaptive_npc.npc.ppo_npc import load_policy
from adaptive_npc.players.base import Player


class LookaheadPlayer(Player):
    def __init__(self, la_cfg, env_cfg, npc_policy_path, predictor_ckpt, seed=0):
        self.cfg, self.spec = la_cfg, EnvSpec(env_cfg)
        model, model_cfg = load_predictor(predictor_ckpt)
        self.phi0 = model.eval()
        self.predictor = OnlinePredictor(model, int(model_cfg["window"]))  # frozen: its NPC-view φ0 window
        self.npc_probs, self.npc_value = policy_fns(load_policy(npc_policy_path))
        oracle = la_cfg["simulator"] == "true"
        self.sim = load_env_model(la_cfg.get("env_model_checkpoint"), oracle_model=oracle, env_cfg=env_cfg)
        self.eps = float(la_cfg["epsilon"])
        self.seed(seed)

    def seed(self, seed):
        self.rng = np.random.default_rng(int(seed))
        self.gen = torch.Generator().manual_seed(int(seed))

    def reset(self):
        self.predictor.reset_round()

    @torch.no_grad()
    def _iop(self, wins):
        return F.softmax(self.phi0(wins), dim=-1)

    def q_values(self, obs, hist):
        """Imagined Q over its 8 candidate actions for this step (advances the φ0 window: call once per step)."""
        if hist:  # previous step's actions: hist entries are (own action, NPC action)
            own, npc = hist[-1]
            self.predictor.observe(npc, own)
        raw = decode_obs(obs, 1, self.spec)
        self.predictor.predict(encode_raw(raw, 0, self.spec)[0].numpy())  # appends the NPC-view row to the window
        c = self.cfg
        G = imagined_returns(self.sim, self.spec, self.npc_probs, self.npc_value, self._iop, raw,
                             self.predictor._pending[None], self.gen, k=int(c["k"]), opp_samples=int(c["opp_samples"]),
                             npc_samples=1, gamma=float(c["gamma"]), imagined_npc=c["imagined_npc"])
        return aggregate_q(G, c["score_agg"])[0]

    def act(self, obs, hist):
        q = self.q_values(obs, hist)
        if self.rng.random() < self.eps:
            return int(self.rng.integers(N_ACTIONS))
        return int(torch.argmax(q))

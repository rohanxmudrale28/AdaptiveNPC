"""NPC adapters for live play: one interface over every NPC version, stepped directly on an ArenaGame.

A LiveNPC mirrors what the evaluation does through NPCEnv (eval/runner.build_npc), minus the gym wrapper:
  reset_round()                       start of every round (learned state persists across rounds)
  act(game) -> action                 chooses the NPC's action; sets `pred` (probabilities over the player's next
                                      action) for versions that predict, else None
  observe(a_npc, a_player, game, r, done)   after the game step
  end_round()
`alpha` (MBOM only) is the posterior over IOP levels for the demo panel.
Methods: a players.yaml archetype / 'random' / 'switcher', 'scripted', 'ppo_none', 'ppo_predicted', 'mbom'.
No pygame here.
"""
from pathlib import Path

import numpy as np

from adaptive_npc.config import load_config
from adaptive_npc.env.observations import encode
from adaptive_npc.models.env_model import raw_state
from adaptive_npc.models.player_model import OnlinePredictor, load_predictor
from adaptive_npc.npc.ppo_npc import PolicyNPC, model_path
from adaptive_npc.players.base import RandomPlayer, Transition
from adaptive_npc.players.scripted import make_player

PPO_MODES = {"ppo_none": "none", "ppo_predicted": "predicted"}
CHECKPOINT_METHODS = tuple(PPO_MODES) + ("mbom",)

TRAIN_HINT = {
    "none": "python -m scripts.train_npc --config configs/npc_ppo.yaml --mode none --seed {seed}",
    "predicted": "python -m scripts.train_npc --config configs/npc_ppo.yaml --mode predicted --seed {seed}",
    "predictor": "python -m scripts.train_predictor --config configs/predictor.yaml",
    "env_model": "python -m scripts.train_env_model --config configs/env_model.yaml",
}


class MissingCheckpointError(FileNotFoundError):
    """A required model file is missing. The message names every missing file and how to create it.
    `missing` = [(path, create-command)]."""

    missing = ()


def require_files(method, items, seed):
    """items: [(path, hint key)]. Raises MissingCheckpointError listing every file that does not exist."""
    missing = [(p, h) for p, h in items if not Path(p).is_file()]
    if missing:
        err_missing = [(p, TRAIN_HINT[h].format(seed=seed)) for p, h in missing]
        lines = [f"Cannot build NPC '{method}': missing checkpoint(s):"]
        lines += [f"  {p}   (create with: {TRAIN_HINT[h].format(seed=seed)})" for p, h in missing]
        err = MissingCheckpointError("\n".join(lines))
        err.missing = err_missing
        raise err


def required_checkpoints(method, npc_cfg, seed):
    """[(path, hint key)] the method needs. Also usable to list/verify checkpoints without loading them."""
    if method not in CHECKPOINT_METHODS:
        return []
    out = [(model_path(npc_cfg["model_dir"], PPO_MODES.get(method, "predicted"), seed),
            PPO_MODES.get(method, "predicted"))]
    if method in ("ppo_predicted", "mbom"):
        out.append((npc_cfg["predictor_checkpoint"], "predictor"))
    if method == "mbom":
        out.append((npc_cfg["imagination"]["env_model_checkpoint"], "env_model"))
    return out


class LiveNPC:
    method = ""
    predicts = False  # True -> `pred` is set by every act()

    def __init__(self):
        self.pred = None
        self.alpha = None

    def reset_round(self):
        pass

    def act(self, game):
        raise NotImplementedError

    def observe(self, a_npc, a_player, game, reward, done):
        pass

    def end_round(self):
        pass


class PlayerNPC(LiveNPC):
    """Archetype / scripted baseline / random: any Player. Keeps the per-round (own, other) history it expects."""

    def __init__(self, method, player):
        super().__init__()
        self.method, self.player = method, player
        self._hist, self._obs = [], None

    def reset_round(self):
        self.player.reset()
        self._hist = []

    def act(self, game):
        self._obs = encode(game, 0)
        return int(self.player.act(self._obs, self._hist))

    def observe(self, a_npc, a_player, game, reward, done):
        self._hist.append((a_npc, a_player))
        if self.player.learns:
            self.player.observe(Transition(self._obs, a_npc, a_player, reward, encode(game, 0), done))


class PPONone(LiveNPC):
    method = "ppo_none"

    def __init__(self, policy, deterministic):
        super().__init__()
        self.model, self.det = policy, bool(deterministic)

    def act(self, game):
        return self.model.act(encode(game, 0), self.det)


class PPOPredicted(LiveNPC):
    """Frozen policy fed the (online-finetuned) GRU prediction of the player's next action: ConditionedEnv's logic."""
    method = "ppo_predicted"
    predicts = True

    def __init__(self, policy, predictor, deterministic):
        super().__init__()
        self.model, self.predictor, self.det = policy, predictor, bool(deterministic)

    def reset_round(self):
        self.predictor.reset_round()

    def act(self, game):
        obs = encode(game, 0)
        self.pred = self.predictor.predict(obs)
        return self.model.act(np.concatenate([obs, self.pred.astype(np.float32)]), self.det)

    def observe(self, a_npc, a_player, game, reward, done):
        self.predictor.observe(a_npc, a_player)


class MBOMLive(LiveNPC):
    """MBOMAgent driven directly (what MBOMEnv does per step): cond -> act -> observe, end_round on the last step."""
    method = "mbom"
    predicts = True

    def __init__(self, agent, policy, deterministic):
        super().__init__()
        self.agent, self.policy, self.det = agent, policy, bool(deterministic)

    def reset_round(self):
        self.agent.reset_round()

    def act(self, game):
        obs = encode(game, 0)
        c = self.agent.cond(obs, raw_state(game))
        self.pred, self.alpha = c, np.array(self.agent.mixer.alpha)
        return int(self.policy.predict(np.concatenate([obs, c]), deterministic=self.det)[0])

    def observe(self, a_npc, a_player, game, reward, done):
        self.agent.observe(a_npc, a_player)
        if done:
            self.agent.end_round()


def build_live_npc(method, ui_cfg, env_cfg, players_cfg, seed):
    """Build the NPC for `method`. `seed` seeds the archetype's randomness; the PPO / MBOM checkpoint seed is
    ui_cfg['npc_seed']. Raises MissingCheckpointError (before loading anything) if a checkpoint is absent."""
    ck_seed = int(ui_cfg["npc_seed"])
    det = bool(ui_cfg["npc_deterministic"])
    if method not in CHECKPOINT_METHODS:
        if method == "random":
            player = RandomPlayer(seed)
        else:
            player = make_player("scripted_npc" if method == "scripted" else method, players_cfg, env_cfg, seed)
        player.seed(seed)
        return PlayerNPC(method, player)

    npc_cfg = load_config(ui_cfg["npc_config"])
    require_files(method, required_checkpoints(method, npc_cfg, ck_seed), ck_seed)
    if method == "ppo_none":
        return PPONone(PolicyNPC(model_path(npc_cfg["model_dir"], "none", ck_seed)), det)
    online = load_config(npc_cfg["predictor_config"])["online"]
    if method == "ppo_predicted":
        model, model_cfg = load_predictor(npc_cfg["predictor_checkpoint"])
        predictor = OnlinePredictor(model, int(model_cfg["window"]), online if npc_cfg["eval"]["predictor_online"] else None)
        return PPOPredicted(PolicyNPC(model_path(npc_cfg["model_dir"], "predicted", ck_seed)), predictor, det)

    # mbom: build_mbom wants a base env only for its observation space, which the agent never uses.
    from adaptive_npc.env.gym_env import NPCEnv
    from adaptive_npc.npc.mbom.agent import build_mbom
    npc_cfg["train"] = {"model_dir": npc_cfg["model_dir"]}
    env, policy = build_mbom(npc_cfg, env_cfg, "mbom", ck_seed, NPCEnv(env_cfg, opponent=RandomPlayer(seed), seed=seed))
    return MBOMLive(env.agent, policy, det)

"""PPO NPC: SB3 PPO on NPCEnv, with three conditioning modes for the observation.

  none       plain PPO on the NPC-view observation
  predicted  GRU predictor's distribution over the player's next action is appended (8 dims)
  oracle     one-hot of the player's TRUE next action is appended (debug upper bound)
"""
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.models.player_model import OnlinePredictor, load_predictor

MODES = ("none", "predicted", "oracle")


class ConditionedEnv(gym.Wrapper):
    """Appends an N_ACTIONS-dim conditioning vector to the NPC observation."""

    def __init__(self, env, mode, predictor=None):
        super().__init__(env)
        if mode not in ("predicted", "oracle"):
            raise ValueError(f"ConditionedEnv mode must be 'predicted' or 'oracle', got {mode!r}")
        if mode == "predicted" and predictor is None:
            raise ValueError("mode 'predicted' needs an OnlinePredictor")
        self.mode, self.predictor = mode, predictor
        base = env.observation_space
        self.observation_space = spaces.Box(
            np.concatenate([base.low, np.zeros(N_ACTIONS, np.float32)]),
            np.concatenate([base.high, np.ones(N_ACTIONS, np.float32)]), dtype=np.float32)

    def _cond(self, obs):
        if self.mode == "oracle":
            c = np.zeros(N_ACTIONS, np.float32)
            c[self.env.unwrapped.peek_opp_action()] = 1.0
        else:
            c = self.predictor.predict(obs).astype(np.float32)
        return np.concatenate([obs, c])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.predictor is not None:
            self.predictor.reset_round()
        return self._cond(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if self.predictor is not None:
            self.predictor.observe(action, info["opp_action"])
        if term or trunc:  # episode over: the next obs is only a placeholder (no peek / predict)
            return np.concatenate([obs, np.full(N_ACTIONS, 1.0 / N_ACTIONS, np.float32)]), r, term, trunc, info
        return self._cond(obs), r, term, trunc, info


class ConditionFactory:
    """Picklable env -> env factory for make_vec_env(wrapper_factory=...). Mode 'none' leaves the env alone.
    `online` None keeps the predictor frozen; otherwise it is the predictor.yaml 'online' finetune block."""

    def __init__(self, mode, predictor_ckpt=None, online=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.ckpt, self.online = mode, predictor_ckpt, online

    def __call__(self, env):
        if self.mode == "none":
            return env
        predictor = None
        if self.mode == "predicted":
            model, model_cfg = load_predictor(self.ckpt)
            predictor = OnlinePredictor(model, int(model_cfg["window"]), self.online)
        return ConditionedEnv(env, self.mode, predictor)


def build_ppo(venv, ppo_cfg, seed):
    net = [int(x) for x in ppo_cfg["net_arch"]]
    return PPO("MlpPolicy", venv, seed=int(seed), device="cpu", verbose=0,
               policy_kwargs={"net_arch": {"pi": net, "vf": net}}, **ppo_cfg["hparams"])


def model_path(model_dir, mode, seed):
    return f"{model_dir}/{mode}_seed{int(seed)}.pt"


# SB3's zip checkpoints fail to load with the installed torch (torch.load on a zip member stream), so the
# trained policy is stored as a plain torch checkpoint and rebuilt as a bare ActorCriticPolicy for inference.
def save_policy(model, path, ppo_cfg):
    torch.save({"state_dict": model.policy.state_dict(), "obs_space": model.observation_space,
                "net_arch": [int(x) for x in ppo_cfg["net_arch"]]}, path)


def load_policy(path):
    """Rebuild the ActorCriticPolicy saved by save_policy (eval mode). Also the start point of learning players."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ck["net_arch"]
    policy = ActorCriticPolicy(ck["obs_space"], spaces.Discrete(N_ACTIONS), lambda _: 0.0,
                               net_arch={"pi": net, "vf": net})
    policy.load_state_dict(ck["state_dict"])
    policy.set_training_mode(False)
    return policy


class PolicyNPC:
    """Frozen PPO policy for inference: act(obs, deterministic) -> action."""

    def __init__(self, path):
        self.policy = load_policy(path)

    def act(self, obs, deterministic=True):
        return int(self.policy.predict(obs, deterministic=deterministic)[0])

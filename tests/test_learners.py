"""M9: ppo_lite, naive / reasoning learners."""
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.models.ppo_lite import PPOLite, action_probs
from adaptive_npc.npc.ppo_npc import load_policy

PPO_CFG = {"lr": 3e-3, "epochs": 4, "minibatch": 32, "clip": 0.2, "ent_coef": 0.0, "vf_coef": 0.5,
           "target_kl": 1.0, "max_grad_norm": 0.5, "gamma": 0.99, "gae_lambda": 0.95, "critic_warmup_updates": 0}


def toy_policy(dim=4, seed=0):
    torch.manual_seed(seed)
    space = spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)
    return ActorCriticPolicy(space, spaces.Discrete(N_ACTIONS), lambda _: 0.0, net_arch={"pi": [16], "vf": [16]})


def run_bandit(ppo, policy, rng, steps=64, good=3):
    """One-step episodes: reward 1 for action `good`, else 0."""
    obs = np.full(4, 0.5, dtype=np.float32)
    for _ in range(steps):
        p = action_probs(policy, obs[None])[0].numpy().astype(np.float64)
        a = int(rng.choice(N_ACTIONS, p=p / p.sum()))
        with torch.no_grad():
            v = float(policy.predict_values(torch.as_tensor(obs)[None])[0, 0])
        ppo.record(obs, a, np.log(p[a]), v, 1.0 if a == good else 0.0, True)
    return ppo.update(0.0)


def test_ppo_lite_learns_bandit():
    policy = toy_policy()
    obs = np.full((1, 4), 0.5, dtype=np.float32)
    before = float(action_probs(policy, obs)[0, 3])
    ppo, rng = PPOLite(policy, PPO_CFG, seed=0), np.random.default_rng(0)
    for _ in range(10):
        run_bandit(ppo, policy, rng)
    after = float(action_probs(policy, obs)[0, 3])
    assert after > before + 0.2, (before, after)


def test_ppo_lite_kl_early_stop():
    policy = toy_policy()
    ppo = PPOLite(policy, {**PPO_CFG, "lr": 0.05, "epochs": 5, "target_kl": 1e-6}, seed=0)
    stats = run_bandit(ppo, policy, np.random.default_rng(0))
    assert stats["epochs_run"] == 1


def test_ppo_lite_critic_warmup_keeps_policy():
    policy = toy_policy()
    obs = np.full((1, 4), 0.5, dtype=np.float32)
    before = action_probs(policy, obs).clone()
    ppo = PPOLite(policy, {**PPO_CFG, "critic_warmup_updates": 1}, seed=0)
    stats = run_bandit(ppo, policy, np.random.default_rng(0))
    assert stats["warmup"]
    assert torch.allclose(before, action_probs(policy, obs), atol=1e-6)


def test_load_policy_roundtrip(tmp_path):
    policy = toy_policy(dim=21)
    path = tmp_path / "p.pt"
    torch.save({"state_dict": policy.state_dict(), "obs_space": policy.observation_space, "net_arch": [16]}, path)
    loaded = load_policy(path)
    x = np.random.default_rng(0).uniform(-1, 1, (5, 21)).astype(np.float32)
    assert torch.allclose(action_probs(policy, x), action_probs(loaded, x))

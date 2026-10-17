"""M9: ppo_lite, naive / reasoning learners."""
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from adaptive_npc.config import load_config
from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.env.match import run_match
from adaptive_npc.env.observations import OBS_DIM
from adaptive_npc.models.player_model import build_model, save_predictor
from adaptive_npc.models.ppo_lite import PPOLite, action_probs
from adaptive_npc.npc.ppo_npc import PolicyNPC, PolicyPlayer, load_policy
from adaptive_npc.players.naive_learner import NaiveLearner
from adaptive_npc.players.reasoning_learner import ReasoningLearner
from adaptive_npc.players.scripted import make_player

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


def save_random_policy(path, dim, seed=0, net=(64, 64)):
    torch.manual_seed(seed)
    space = spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)
    pol = ActorCriticPolicy(space, spaces.Discrete(N_ACTIONS), lambda _: 0.0, net_arch={"pi": list(net), "vf": list(net)})
    torch.save({"state_dict": pol.state_dict(), "obs_space": space, "net_arch": list(net)}, path)
    return path


def save_random_npc_model(path, seed=0):
    torch.manual_seed(seed)
    cfg = {"hidden": 16, "layers": 1, "window": 8}
    save_predictor(path, build_model(cfg), cfg)
    return path


def scores(results):
    """Learner (slot 1) score per round: win 1, draw 1/2, loss 0."""
    return np.array([1.0 if r["winner"] == 1 else 0.5 if r["winner"] == -1 else 0.0 for r in results])


def test_naive_learner_win_rate_rises(tmp_path):
    """Random-init naive learner vs a frozen rusher: last third of the match beats the first third."""
    env_cfg, players_cfg = load_config("configs/env.yaml"), load_config("configs/players.yaml")
    path = save_random_policy(tmp_path / "naive.pt", OBS_DIM)
    learner = NaiveLearner(path, {**PPO_CFG, "lr": 3e-3, "ent_coef": 0.01, "target_kl": 0.03}, 64, seed=0)
    res = run_match(env_cfg, make_player("rusher", players_cfg, env_cfg, 5), learner, n_rounds=150, seed=3)
    s, ret = scores(res), -np.array([r["ret_npc"] for r in res])
    assert learner.n_updates > 0
    assert s[100:].mean() > s[:50].mean() + 0.1, (s[:50].mean(), s[100:].mean())
    assert ret[100:].mean() > ret[:50].mean()


def test_frozen_learner_does_not_change(tmp_path):
    env_cfg, players_cfg = load_config("configs/env.yaml"), load_config("configs/players.yaml")
    path = save_random_policy(tmp_path / "naive.pt", OBS_DIM)
    learner = NaiveLearner(path, PPO_CFG, 16, learn=False, seed=0)
    x = np.random.default_rng(0).uniform(-1, 1, (8, OBS_DIM)).astype(np.float32)
    before = learner.policy_probs(x).clone()
    run_match(env_cfg, make_player("rusher", players_cfg, env_cfg, 5), learner, n_rounds=3, seed=3)
    assert learner.n_updates == 0 and torch.equal(before, learner.policy_probs(x))


def test_reasoning_learner_predicts_npc_above_chance(tmp_path):
    env_cfg, players_cfg = load_config("configs/env.yaml"), load_config("configs/players.yaml")
    pol = save_random_policy(tmp_path / "reason.pt", OBS_DIM + N_ACTIONS)
    gru = save_random_npc_model(tmp_path / "npc_model.pt")
    online = {"lr": 3e-3, "every": 10, "grad_steps": 3, "buffer": 256, "min_buffer": 32}
    learner = ReasoningLearner(pol, gru, PPO_CFG, 64, online, seed=0)
    run_match(env_cfg, make_player("rusher", players_cfg, env_cfg, 5), learner, n_rounds=20, seed=3)
    hits = np.array(learner.pred_hits)
    assert len(hits) > 200
    assert hits[len(hits) // 2:].mean() > 1.0 / N_ACTIONS + 0.1, hits[len(hits) // 2:].mean()
    assert learner.n_updates > 0


def test_learner_match_is_deterministic(tmp_path):
    env_cfg, players_cfg = load_config("configs/env.yaml"), load_config("configs/players.yaml")
    pol = save_random_policy(tmp_path / "reason.pt", OBS_DIM + N_ACTIONS)
    gru = save_random_npc_model(tmp_path / "npc_model.pt")
    online = {"lr": 3e-3, "every": 10, "grad_steps": 3, "buffer": 256, "min_buffer": 32}
    runs = []
    for _ in range(2):
        learner = ReasoningLearner(pol, gru, PPO_CFG, 32, online, seed=7)
        runs.append(run_match(env_cfg, make_player("mixer", players_cfg, env_cfg, 5), learner, n_rounds=4, seed=3))
    assert runs[0] == runs[1]


def test_policy_player_matches_policy_npc_and_plays(tmp_path):
    """PolicyPlayer greedy == PolicyNPC greedy; the predicted-conditioned variant plays a match from slot 1."""
    env_cfg, players_cfg = load_config("configs/env.yaml"), load_config("configs/players.yaml")
    plain = save_random_policy(tmp_path / "plain.pt", OBS_DIM, net=(16,))
    player, npc = PolicyPlayer(plain), PolicyNPC(plain)
    for x in np.random.default_rng(0).uniform(-1, 1, (20, OBS_DIM)).astype(np.float32):
        assert player.act(x, []) == npc.act(x, True)
    cond = save_random_policy(tmp_path / "cond.pt", OBS_DIM + N_ACTIONS, net=(16,))
    gru = save_random_npc_model(tmp_path / "gru.pt")
    res = run_match(env_cfg, make_player("rusher", players_cfg, env_cfg, 5), PolicyPlayer(cond, gru), n_rounds=2, seed=3)
    assert len(res) == 2


def test_load_policy_roundtrip(tmp_path):
    policy = toy_policy(dim=21)
    path = tmp_path / "p.pt"
    torch.save({"state_dict": policy.state_dict(), "obs_space": policy.observation_space, "net_arch": [16]}, path)
    loaded = load_policy(path)
    x = np.random.default_rng(0).uniform(-1, 1, (5, 21)).astype(np.float32)
    assert torch.allclose(action_probs(policy, x), action_probs(loaded, x))

import numpy as np
import pytest

from adaptive_npc.config import load_config
from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.env.observations import OBS_DIM
from adaptive_npc.env.vec_env import make_vec_env
from adaptive_npc.eval.metrics import aggregate, mean_ci, summarize_rounds
from adaptive_npc.eval.runner import run_grid
from adaptive_npc.models.player_model import build_model, save_predictor
from adaptive_npc.npc.ppo_npc import ConditionFactory, build_ppo, model_path, save_policy
from adaptive_npc.players.pool import PoolFactory, PoolPlayer
from adaptive_npc.players.scripted import make_player

ENV = load_config("configs/env.yaml")
PLAYERS = load_config("configs/players.yaml")
CFG = load_config("configs/experiments/npc_smoke.yaml")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("pred") / "p.pt"
    mc = {"hidden": 8, "layers": 1, "window": 8}
    save_predictor(path, build_model(mc), mc)
    return str(path)


def test_oracle_conditioning_is_the_true_next_action():
    env = ConditionFactory("oracle")(NPCEnv(ENV, opponent=make_player("mixer", PLAYERS, ENV, 3), seed=1))
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM + N_ACTIONS,)
    for _ in range(40):
        cond = obs[OBS_DIM:]
        assert cond.sum() == 1.0
        obs, _, done, _, info = env.step(env.action_space.sample())
        assert cond.argmax() == info["opp_action"]
        if done:
            obs, _ = env.reset()


def test_peek_does_not_change_trajectories():
    def run(peek):
        env = NPCEnv(ENV, opponent=make_player("mixer", PLAYERS, ENV, 3), seed=7)
        env.reset()
        acts = []
        for i in range(30):
            if peek:
                env.peek_opp_action()
            _, _, done, _, info = env.step(i % N_ACTIONS)
            acts.append(info["opp_action"])
            if done:
                env.reset()
        return acts
    assert run(True) == run(False)


def test_predicted_conditioning_is_a_distribution(ckpt):
    env = ConditionFactory("predicted", ckpt)(NPCEnv(ENV, seed=2))
    obs, _ = env.reset()
    for _ in range(10):
        cond = obs[OBS_DIM:]
        assert cond.shape == (N_ACTIONS,) and abs(cond.sum() - 1.0) < 1e-5
        obs, _, done, _, _ = env.step(0)
        if done:
            break


def test_none_mode_leaves_env_unwrapped():
    env = NPCEnv(ENV, seed=0)
    assert ConditionFactory("none")(env) is env


def test_pool_player_samples_members_and_is_seeded():
    def names(seed):
        p = PoolPlayer(["rusher", "camper", "mixer"], PLAYERS, ENV, seed)
        out = []
        for _ in range(30):
            p.reset()
            out.append(p.current_name)
        return out
    assert names(4) == names(4)
    assert len(set(names(4))) == 3


def test_metrics():
    rounds = [{"winner": 0, "steps": 10, "ret_npc": 1.0}, {"winner": -1, "steps": 20, "ret_npc": 0.0},
              {"winner": 1, "steps": 30, "ret_npc": -1.0}, {"winner": 0, "steps": 40, "ret_npc": 1.0}]
    s = summarize_rounds(rounds)
    assert s["win_rate"] == pytest.approx(0.625) and s["wins"] == 0.5 and s["mean_steps"] == 25
    m, hw = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and hw == pytest.approx(2.4841, abs=1e-3)  # t(0.975, 2) * 1/sqrt(3)
    assert np.isnan(mean_ci([1.0])[1])
    agg = aggregate([{"method": "a", "opponent": "x", "win_rate": 0.5, "mean_return": 0.0},
                     {"method": "a", "opponent": "x", "win_rate": 0.7, "mean_return": 1.0}])
    assert agg[0]["n_seeds"] == 2 and agg[0]["win_rate_mean"] == pytest.approx(0.6)


@pytest.mark.parametrize("mode", ["none", "predicted", "oracle"])
def test_tiny_ppo_train_and_eval_grid(tmp_path, ckpt, mode):
    cfg = {**CFG, "predictor_checkpoint": ckpt, "online": load_config("configs/predictor.yaml")["online"],
           "train": {**CFG["train"], "model_dir": str(tmp_path)},
           "eval": {**CFG["eval"], "rounds": 2, "predictor_online": mode == "predicted"}}
    venv = make_vec_env(ENV, n_envs=2, seed=0, opponent_factory=PoolFactory(["rusher", "camper"], PLAYERS, ENV),
                        wrapper_factory=ConditionFactory(mode, ckpt))
    model = build_ppo(venv, CFG["ppo"], 0)
    model.learn(256)
    save_policy(model, model_path(tmp_path, mode, 0), CFG["ppo"])
    rows, summary = run_grid(cfg, ENV, PLAYERS, tmp_path / "out", methods=[f"ppo_{mode}", "scripted"],
                             opponents=["rusher"], seeds=[0])
    assert len(rows) == 2 and (tmp_path / "out" / "metrics.csv").exists()
    assert (tmp_path / "out" / f"ppo_{mode}" / "rusher" / "seed_0" / "metrics.csv").exists()
    assert all(r["rounds"] == 2 for r in rows)

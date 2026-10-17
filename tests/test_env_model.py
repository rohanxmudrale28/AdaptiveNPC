"""M7: env model Γ vs the true simulator (no checkpoints needed: Γ is trained on the fly from random play)."""
import numpy as np
import pytest
import torch

from adaptive_npc.config import load_config
from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.eval.env_model_eval import evaluate
from adaptive_npc.models.env_model import (CHANNEL_IDX, STATE_DIM, EnvModel, EnvSpec, TrueSimulator, features,
                                           load_env_model, loss_fn, raw_state, rollout, save_env_model, targets)
from adaptive_npc.seeding import seed_everything

ENV = load_config("configs/env.yaml")
SPEC = EnvSpec(ENV)


def random_rounds(n_rounds, seed):
    """Random-vs-random rounds through the real core -> transition arrays in time order."""
    rng = np.random.default_rng(seed)
    g = ArenaGame(ENV, rng=np.random.default_rng(seed + 1))
    cols = {k: [] for k in ("state", "next_state", "a_npc", "a_opp", "reward", "done", "ep")}
    for ep in range(n_rounds):
        g.reset()
        done = False
        while not done:
            s = raw_state(g)
            a0, a1 = (int(a) for a in rng.integers(N_ACTIONS, size=2))
            r, done = g.step(a0, a1)
            for k, v in zip(cols, (s, raw_state(g), a0, a1, r, done, ep)):
                cols[k].append(v)
    d = {k: np.array(v) for k, v in cols.items()}
    d["state"], d["next_state"] = d["state"].astype(np.int16), d["next_state"].astype(np.int16)
    return d


@pytest.fixture(scope="module")
def trained():
    seed_everything(0)
    train, held = random_rounds(250, 10), random_rounds(40, 99)
    model = EnvModel(SPEC, [128, 128])
    x = features(SPEC, train["state"], train["a_npc"], train["a_opp"])
    tg = targets(SPEC, train["state"], train["next_state"], train["reward"])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(25):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            b = perm[i:i + 256]
            loss, _ = loss_fn(SPEC, model(x[b]), tg[0][b], tg[1][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), held


def test_true_simulator_replays_the_core_exactly():
    """oracle_model debug flag: the TrueSimulator reproduces real game trajectories state for state."""
    d = random_rounds(5, 3)
    sim = TrueSimulator(ENV)
    nxt, rew, done = sim.step(d["state"], d["a_npc"], d["a_opp"])
    assert (nxt == d["next_state"]).all()
    assert np.allclose(rew, d["reward"]) and (done == d["done"]).all()
    # open-loop rollout of a whole first round equals the recorded trajectory
    m = d["ep"] == 0
    states, rewards, _ = rollout(sim, d["state"][m][:1], d["a_npc"][m][None], d["a_opp"][m][None])
    assert (states[0, 1:] == d["next_state"][m]).all() and np.allclose(rewards[0], d["reward"][m])


def test_oracle_flag_loads_simulator_without_checkpoint():
    assert isinstance(load_env_model(None, oracle_model=True, env_cfg=ENV), TrueSimulator)


def test_state_roundtrip_and_snapping(trained):
    model, held = trained
    nxt, r, done = model.step(held["state"][:50], held["a_npc"][:50], held["a_opp"][:50])
    assert nxt.shape == (50, STATE_DIM) and nxt.dtype == np.int16
    assert (nxt >= SPEC.lo).all() and (nxt <= SPEC.hi).all()  # always a legal integer state
    assert (nxt[:, -1] == held["state"][:50, -1] + 1).all()  # t advances deterministically


def test_one_step_beats_copy_baseline_on_held_out_transitions(trained):
    model, held = trained
    rows = {(r["horizon"], r["feature"]): r for r in evaluate(model, held, 0.05)}
    one = rows[(1, "all_state")]
    assert one["exact"] > 0.75 and one["exact"] > one["copy_exact"] + 0.5, one  # tiny model, 36k transitions
    for ch in ("hp_npc", "hp_opp", "stun_npc", "stun_opp", "cd_npc", "cd_opp"):
        assert rows[(1, ch)]["exact"] > 0.98, (ch, rows[(1, ch)])
    assert rows[(1, "reward")]["exact"] > 0.97
    assert rows[(2, "all_state")]["exact"] > rows[(2, "all_state")]["copy_exact"] + 0.4


def test_rollouts_match_the_true_simulator(trained):
    """Γ and the true simulator, same start states and action sequences: per-step full-state agreement."""
    model, held = trained
    rng = np.random.default_rng(5)
    starts = np.where(~held["done"])[0][::7][:300]
    k = 3
    keep = np.array([i for i in starts if i + k < len(held["state"]) and (held["ep"][i + k] == held["ep"][i])])
    s0 = held["state"][keep]
    a0, a1 = rng.integers(N_ACTIONS, size=(2, len(keep), k))
    truth, true_r, _ = rollout(TrueSimulator(ENV), s0, a0, a1)
    pred, pred_r, _ = rollout(model, s0, a0, a1)
    idx = list(CHANNEL_IDX)
    agree = [float((pred[:, j][:, idx] == truth[:, j][:, idx]).all(1).mean()) for j in range(1, k + 1)]
    assert agree[0] > 0.75 and agree[1] > 0.6 and agree[2] > 0.45, agree  # degrades with k, but stays usable
    assert np.abs(pred_r - true_r).mean() < 0.1


def test_checkpoint_roundtrip(trained, tmp_path):
    model, held = trained
    path = tmp_path / "g.pt"
    save_env_model(path, model, ENV)
    loaded = load_env_model(path)
    a = model.step(held["state"][:20], held["a_npc"][:20], held["a_opp"][:20])
    b = loaded.step(held["state"][:20], held["a_npc"][:20], held["a_opp"][:20])
    assert (a[0] == b[0]).all() and np.allclose(a[1], b[1])


def test_random_action_augmentation_is_exact_pool_limited_and_sized():
    from adaptive_npc.models.env_model_data import augment_random_actions
    pool = random_rounds(20, 4)["state"]
    aug = augment_random_actions(ENV, pool, n_base=3000, fraction=0.4, two_step_share=0.5, hard_weight=2.0, seed=11)
    n = len(aug["state"])
    assert abs(n / (n + 3000) - 0.4) < 0.01
    nxt, rew, _ = TrueSimulator(ENV).step(aug["state"], aug["a_npc"], aug["a_opp"])  # labels are exact
    assert (nxt == aug["next_state"]).all() and np.allclose(rew, aug["reward"])
    pool_set = {tuple(r) for r in pool}
    n_pool_starts = sum(tuple(r) in pool_set for r in aug["state"])
    assert 0.5 * n <= n_pool_starts < n  # one-step starts + first steps of chains come from the pool; chain tails do not

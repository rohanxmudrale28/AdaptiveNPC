"""M8: MBOM components (no checkpoints needed unless marked)."""
import csv
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from adaptive_npc.config import load_config
from adaptive_npc.env.actions import Action
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.observations import encode
from adaptive_npc.models.env_model import STATE_DIM, EnvSpec, TrueSimulator, load_env_model, raw_state
from adaptive_npc.models.player_model import FEAT_DIM, PlayerPredictor
from adaptive_npc.eval.runner import run_grid
from adaptive_npc.npc.mbom.agent import MBOMAgent, parse_method
from adaptive_npc.npc.mbom.imagination import aggregate_q, br_targets, imagined_returns
from adaptive_npc.npc.mbom.iops import IOPSet
from adaptive_npc.npc.mbom.mixer import BayesMixer
from adaptive_npc.npc.mbom.state_obs import encode_raw

ENV = load_config("configs/env.yaml")
SPEC = EnvSpec(ENV)


def random_legal_states(n, seed):
    """Uniform over each channel's legal range (positions may coincide; encode does not care)."""
    rng = np.random.default_rng(seed)
    return rng.integers(SPEC.lo, SPEC.hi.astype(np.int64) + 1, size=(n, STATE_DIM)).astype(np.int16)


def set_game(g, s):
    g.x, g.y = [int(s[0]), int(s[2])], [int(s[1]), int(s[3])]
    g.face, g.hp = [int(s[4]), int(s[5])], [int(s[6]), int(s[7])]
    g.sta, g.cd, g.stun = [int(s[8]), int(s[9])], [int(s[10]), int(s[11])], [int(s[12]), int(s[13])]
    g.t = int(s[14])


def test_encode_raw_matches_encode_both_views():
    raw = random_legal_states(1000, 0)
    g = ArenaGame(ENV, rng=np.random.default_rng(0))
    for who in (0, 1):
        ref = []
        for s in raw:
            set_game(g, s)
            assert np.array_equal(raw_state(g), s)
            ref.append(encode(g, who))
        got = encode_raw(raw, who, SPEC).numpy()
        assert got.dtype == np.float32
        assert np.array_equal(got, np.stack(ref))


def test_encode_raw_on_real_rollout_states():
    g = ArenaGame(ENV, rng=np.random.default_rng(3))
    rng = np.random.default_rng(4)
    states, ref = [], []
    for _ in range(5):
        g.reset()
        done = False
        while not done:
            states.append(raw_state(g))
            ref.append(encode(g, 0))
            _, done = g.step(*(int(a) for a in rng.integers(8, size=2)))
    got = encode_raw(torch.as_tensor(np.stack(states)), 0, ENV)  # accepts a tensor and a raw env cfg
    assert np.array_equal(got.numpy(), np.stack(ref))


# ---- mixer (synthetic: 3 fixed categorical "IOPs", actions sampled from one generating level) ----

def synthetic_levels(peak=0.4):
    """Level m puts `peak` on action 2m and spreads the rest uniformly: overlapping but distinct."""
    d = np.full((3, 8), (1 - peak) / 7)
    for m in range(3):
        d[m, 2 * m] = peak
    return d


def run_mixer(mixer, levels, gen_seq, seed=0):
    rng = np.random.default_rng(seed)
    alphas = []
    for g in gen_seq:
        a = rng.choice(8, p=levels[g])
        mixer.update(levels[:, a])
        alphas.append(mixer.alpha)
    return np.array(alphas)


def stable_from(argmaxes, level):
    """First index from which `level` leads for good (len if it never does)."""
    off = np.nonzero(argmaxes != level)[0]
    return int(off.max()) + 1 if len(off) else 0


@pytest.mark.parametrize("peak,max_median_lag,max_lag", [(0.4, 60, 120), (0.6, 30, 40)])
def test_mixer_converges_to_generating_level_and_follows_a_switch(peak, max_median_lag, max_lag):
    """Default mixer settings, 20 seeds. The switch lag (steps until the new level leads for good) is dominated by
    the sticky prior (moving average of posteriors, floored at prior_floor), not by the evidence: measured median /
    max 49.5 / 98 steps for peak 0.4 and 20 / 26 for peak 0.6. Initial convergence: stable lead by step 17."""
    levels, lags = synthetic_levels(peak), []
    for seed in range(20):
        mixer = BayesMixer(3, decay=0.9, temperature=0.5, prior_avg=0.9, eps=1e-3, prior_floor=0.01)
        al = run_mixer(mixer, levels, [1] * 300 + [2] * 300, seed)
        assert stable_from(al[:300].argmax(1), 1) <= 50, f"seed {seed}: level 1 not leading for good by step 50"
        # steady state sits a little under the cap (prior floor + overlapping likelihoods: measured 0.71-0.77)
        assert al[200:300, 1].mean() > 0.85 * mixer.alpha_cap
        lag = stable_from(al[300:].argmax(1), 2)
        assert lag < 300, f"seed {seed}: never switched"
        assert al[500:, 2].mean() > 0.85 * mixer.alpha_cap
        lags.append(lag)
    assert np.median(lags) <= max_median_lag and max(lags) <= max_lag, f"switch lags {sorted(lags)}"


def test_mixer_alpha_cap():
    mixer = BayesMixer(3, temperature=0.5)
    assert mixer.alpha_cap == pytest.approx(np.e ** 2 / (np.e ** 2 + 2))  # ≈ 0.787
    assert 0.78 < mixer.alpha_cap < 0.79
    al = run_mixer(mixer, synthetic_levels(peak=0.9), [0] * 500)
    assert al.max() <= mixer.alpha_cap + 1e-12
    assert al[-1, 0] == pytest.approx(mixer.alpha_cap, abs=1e-3)
    assert np.allclose(al.sum(1), 1.0)


def test_mixer_mode_flags():
    levels = synthetic_levels()
    uni = run_mixer(BayesMixer(3, mode="uniform"), levels, [1] * 100)
    assert np.allclose(uni, 1 / 3)
    for m in range(3):
        fixed = run_mixer(BayesMixer(3, mode="fixed", fixed_level=m), levels, [(m + 1) % 3] * 100)
        assert np.all(fixed == np.eye(3)[m])
    mixer = BayesMixer(3, mode="fixed", fixed_level=2)
    assert np.allclose(mixer.mix(levels), levels[2])
    with pytest.raises(ValueError):
        BayesMixer(3, mode="fixed")
    with pytest.raises(ValueError):
        BayesMixer(3, mode="softmax")


def test_mixer_mix_is_alpha_weighted_and_reset_restores_uniform():
    levels = synthetic_levels()
    mixer = BayesMixer(3)
    run_mixer(mixer, levels, [0] * 50)
    assert np.allclose(mixer.mix(levels), mixer.alpha @ levels)
    assert mixer.mix(levels).sum() == pytest.approx(1.0)
    mixer.reset()
    assert np.allclose(mixer.alpha, 1 / 3)


# ---- imagination (eq. 5): known best responses in hand-built states ----

GAMMA = 0.99
#          x0 y0 x1 y1 f0 f1 hp0 hp1 sta0 sta1 cd0 cd1 stun0 stun1 t
# NPC at (4,4) facing right, opponent adjacent at (5,4) facing left (each in the other's front tile).
STUNNED = [4, 4, 5, 4, 3, 2, 1, 5, 6, 6, 0, 0, 2, 0, 10]  # NPC stunned with 1 HP: attacking wins at once
DUEL = [4, 4, 5, 4, 3, 2, 1, 1, 6, 6, 0, 0, 0, 0, 10]     # both 1 HP, NPC always attacks: block (parry), then hit


def stub_npc(action, value=0.0):
    """NPC model that always plays `action`, with a constant value."""
    def probs(x):
        return F.one_hot(torch.full((x.shape[0],), action), 8).float()

    def val(x):
        return torch.full((x.shape[0],), float(value))
    return probs, val


def uniform_iop(wins):
    return torch.full((wins.shape[0], 8), 1 / 8)


def brute_force_q(state, npc_action, k=2, value=0.0):
    """Exact max over all 8^k opponent continuations with the true rules and a constant-action NPC."""
    sim = TrueSimulator(ENV)
    q = np.full(8, -np.inf)
    for c in range(8):
        for cont in itertools.product(range(8), repeat=k):
            s, g, alive = np.array([state], np.int16), 0.0, True
            for t, a in enumerate((c,) + cont):
                s, r, d = sim.step(s, [npc_action], [a])
                g += GAMMA ** t * alive * -float(r[0])
                alive = alive and not d[0]
            g += GAMMA ** (k + 1) * alive * -value
            q[c] = max(q[c], g)
    return q


def imagine(model, state, npc_action, opp_samples, value=0.0, seed=0, **kw):
    probs, val = stub_npc(npc_action, value)
    raw = torch.tensor([state])
    win = torch.zeros(1, 8, FEAT_DIM)
    return imagined_returns(model, SPEC, probs, val, uniform_iop, raw, win, torch.Generator().manual_seed(seed),
                            k=2, opp_samples=opp_samples, gamma=GAMMA, **kw)


@pytest.mark.parametrize("state,npc_action,best", [(STUNNED, Action.IDLE, Action.ATTACK),
                                                   (DUEL, Action.ATTACK, Action.BLOCK)])
def test_imagination_finds_known_best_response_true_simulator(state, npc_action, best):
    truth = brute_force_q(state, npc_action)
    assert truth.argmax() == best  # the hand-built case is what we think it is
    q = aggregate_q(imagine(TrueSimulator(ENV), state, npc_action, opp_samples=64), "max")[0].numpy()
    assert q.argmax() == best
    assert np.all(q <= truth + 1e-5)  # sampled continuations can only miss the best one
    assert q[best] == pytest.approx(truth[best], abs=1e-5)
    assert br_targets(torch.tensor(q)[None], 0.1)[0].argmax() == best


@pytest.mark.skipif(not Path("models/env_model.pt").exists(), reason="needs the trained Γ checkpoint")
@pytest.mark.parametrize("state,npc_action,best", [(STUNNED, Action.IDLE, Action.ATTACK),
                                                   (DUEL, Action.ATTACK, Action.BLOCK)])
def test_imagination_finds_known_best_response_learned_gamma(state, npc_action, best):
    gamma_model = load_env_model("models/env_model.pt")
    q = aggregate_q(imagine(gamma_model, state, npc_action, opp_samples=64), "max")[0].numpy()
    assert q.argmax() == best
    assert q[best] == pytest.approx(brute_force_q(state, npc_action)[best], abs=1e-4)


def test_done_mask_zeroes_rewards_and_bootstrap_after_terminal():
    # attacking ends the round at t = 0 (opponent wins): every continuation must score exactly r_0, even though the
    # simulator keeps stepping the finished state and the NPC's value is large
    G = imagine(TrueSimulator(ENV), STUNNED, Action.IDLE, opp_samples=16, value=5.0)
    win_r = -(SPEC.r_loss - abs(ENV["reward"]["damage_taken"]))  # opponent's reward for the killing hit
    assert torch.allclose(G[0, Action.ATTACK], torch.tensor(win_r, dtype=torch.float32))
    # a non-terminal candidate keeps the bootstrap -γ^{k+1} V
    assert G[0, Action.IDLE].min() < -4.0


def test_score_agg_and_imagined_npc_flags():
    sim = TrueSimulator(ENV)
    probs = lambda x: torch.full((x.shape[0], 8), 1 / 8)  # noqa: E731 (stochastic NPC)
    val = lambda x: torch.zeros(x.shape[0])  # noqa: E731
    raw, win = torch.tensor([DUEL, STUNNED]), torch.zeros(2, 8, FEAT_DIM)
    run = lambda **kw: imagined_returns(sim, SPEC, probs, val, uniform_iop, raw, win,  # noqa: E731
                                        torch.Generator().manual_seed(3), k=2, opp_samples=8, gamma=GAMMA, **kw)
    G = run(npc_samples=4)
    assert G.shape == (2, 8, 8, 4)
    assert torch.equal(G, run(npc_samples=4))  # same generator seed -> same returns
    assert torch.equal(G, run(npc_samples=4, chunk_rows=64))  # independent of chunking
    q_max, q_mean = aggregate_q(G, "max"), aggregate_q(G, "mean")
    assert torch.all(q_max >= q_mean - 1e-6)
    assert torch.allclose(q_max, G.mean(3).max(2).values)
    assert run(imagined_npc="greedy", npc_samples=4).shape == (2, 8, 8, 1)
    with pytest.raises(ValueError):
        aggregate_q(G, "median")
    with pytest.raises(ValueError):
        run(imagined_npc="softmax")


def test_npc_samples_are_shared_across_candidates():
    # with an NPC whose action does not depend on the state, every candidate sees the same NPC action at t = 0;
    # check it through the returns: STUNNED forces the NPC idle, so use DUEL with an NPC that attacks w.p. 1/2
    sim = TrueSimulator(ENV)
    probs = lambda x: torch.tensor([[0.5, 0, 0, 0, 0, 0.5, 0, 0]]).expand(x.shape[0], 8)  # noqa: E731
    val = lambda x: torch.zeros(x.shape[0])  # noqa: E731
    G = imagined_returns(sim, SPEC, probs, val, uniform_iop, torch.tensor([DUEL]), torch.zeros(1, 8, FEAT_DIM),
                         torch.Generator().manual_seed(0), k=2, opp_samples=4, npc_samples=8, gamma=GAMMA)
    # candidate 'attack': NPC attacks -> both die (0); NPC idles -> opponent kills it (+1.1). Same per NPC sample
    # for every opponent continuation, and the 'idle' candidate loses exactly where the NPC attacked.
    att, idle = G[0, Action.ATTACK], G[0, Action.IDLE]
    npc_attacked = att[0] == 0
    assert torch.all(att == att[0:1])  # identical across continuations (round over at t = 0)
    assert npc_attacked.any() and (~npc_attacked).any()
    assert torch.all(idle[:, npc_attacked] < -1.0)  # NPC's hit kills the idle opponent at t = 0


def test_iop_finetune_moves_toward_targets():
    model = PlayerPredictor(hidden=16)
    iops = IOPSet(model, n_levels=3, lr=1e-2, grad_steps=20, buffer=8)
    wins = torch.randn(32, 8, FEAT_DIM)
    for i in range(10):
        iops.record(wins[i], torch.tensor(STUNNED), i % 8)
    w, r, a = iops.buffer_tensors()
    assert w.shape == (8, 8, FEAT_DIM) and r.shape == (8, 15) and a.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]
    target = torch.zeros(32, 8)
    target[:, 6] = 1.0
    before = iops.probs(1, wins)[:, 6].mean()
    iops.finetune(1, wins, target)
    assert iops.probs(1, wins)[:, 6].mean() > before + 0.3
    assert torch.allclose(iops.probs(0, wins), iops.probs(2, wins))  # other levels untouched
    iops.finetune(0, wins, torch.full((32,), 3))
    assert iops.probs(0, wins)[:, 3].mean() > 0.5
    assert iops.predict_all(wins[0]).shape == (3, 8)


def test_true_simulator_step_t_matches_step():
    sim, raw = TrueSimulator(ENV), random_legal_states(200, 5)
    rng = np.random.default_rng(6)
    a0, a1 = rng.integers(8, size=200), rng.integers(8, size=200)
    n, r, d = sim.step(raw, a0, a1)
    nt, rt, dt = sim.step_t(torch.from_numpy(raw.astype(np.int64)), torch.from_numpy(a0), torch.from_numpy(a1))
    assert np.array_equal(nt.numpy(), n) and np.array_equal(rt.numpy(), r) and np.array_equal(dt.numpy(), d)


# ---- agent / ablations (small random-init policy and predictor, true simulator; no checkpoints) ----

def tiny_agent(ablation="none", level=None, cond_input="mix", update_every=10):
    from gymnasium import spaces
    from stable_baselines3.common.policies import ActorCriticPolicy
    torch.manual_seed(0)
    policy = ActorCriticPolicy(spaces.Box(-1, 1, (29,), np.float32), spaces.Discrete(8), lambda _: 0.0,
                               net_arch={"pi": [16], "vf": [16]})
    mb = {"n_levels": 3, "update_every": update_every, "buffer": 32, "lr": 1e-3, "grad_steps": 2,
          "warmup_grad_steps": 3, "cond_input": cond_input}
    im = {"k": 2, "opp_samples": 2, "npc_samples": 1, "score_agg": "max", "imagined_npc": "stochastic",
          "tau_br": 0.1, "gamma": 0.99, "chunk_rows": 4096}
    mix = {"decay": 0.9, "temperature": 0.5, "prior_avg": 0.9, "eps": 1e-3, "prior_floor": 0.01}
    return MBOMAgent(mb, im, mix, SPEC, PlayerPredictor(hidden=16), 8, policy, TrueSimulator(ENV), ablation, level,
                     seed=0)


def drive(agent, n_steps, seed=0):
    """Feed the agent n_steps real-looking steps (random legal states, random actions); rounds of 15 steps."""
    rng = np.random.default_rng(seed)
    raw = random_legal_states(n_steps, seed + 100)
    raw[:, 6:8] = np.maximum(raw[:, 6:8], 1)  # alive
    obs = encode_raw(raw, 0, SPEC).numpy()
    conds = []
    for i in range(n_steps):
        conds.append(agent.cond(obs[i], raw[i]))
        agent.observe(int(rng.integers(8)), int(rng.integers(8)))
        if (i + 1) % 15 == 0:
            agent.end_round()
    return np.array(conds)


def test_parse_method():
    assert parse_method("mbom") == ("none", None)
    assert parse_method("mbom_no_imagination") == ("no_imagination", None)
    assert parse_method("mbom_single_level2") == ("single_level", 2)
    with pytest.raises(ValueError):
        parse_method("mbom_bogus")


def count_imagination(monkeypatch, fail=False):
    import adaptive_npc.npc.mbom.agent as agent_mod
    calls = []
    real = agent_mod.imagined_returns

    def spy(*a, **kw):
        if fail:
            raise AssertionError("imagination must not run for this ablation")
        calls.append(a[4])  # iop_prev
        return real(*a, **kw)
    monkeypatch.setattr(agent_mod, "imagined_returns", spy)
    return calls


def test_no_imagination_never_imagines_and_feeds_phi0(monkeypatch):
    count_imagination(monkeypatch, fail=True)
    agent = tiny_agent("no_imagination")
    conds = drive(agent, 30)
    assert agent.n_updates == 3
    assert np.allclose(agent.mixer.alpha, [1, 0, 0])
    assert all(np.allclose([r[f"alpha_l{m}_mean"] for m in range(3)], [1, 0, 0]) for r in agent.round_log)
    assert conds.shape == (30, 8) and np.allclose(conds.sum(1), 1, atol=1e-5)


def test_random_finetune_uses_uniform_random_labels(monkeypatch):
    count_imagination(monkeypatch, fail=True)
    agent = tiny_agent("random_finetune")
    seen = []
    real = agent.iops.finetune
    agent.iops.finetune = lambda m, w, t, steps=None: (seen.append((m, t.clone())), real(m, w, t, steps))[1]
    drive(agent, 200)
    labels = torch.cat([t for m, t in seen if m >= 1])
    assert labels.dtype == torch.long and {m for m, _ in seen} == {0, 1, 2}
    freq = torch.bincount(labels, minlength=8).float() / len(labels)
    assert torch.all((freq - 1 / 8).abs() < 0.03)


@pytest.mark.parametrize("level", [1, 2])
def test_single_level_builds_levels_up_to_m_and_fixes_alpha(monkeypatch, level):
    calls = count_imagination(monkeypatch)
    agent = tiny_agent("single_level", level)
    drive(agent, 20)
    assert len(calls) == 2 * level  # 2 updates x levels 1..m
    assert np.allclose(agent.mixer.alpha, np.eye(3)[level])


def test_full_mbom_and_uniform_mix(monkeypatch):
    calls = count_imagination(monkeypatch)
    agent = tiny_agent("none")
    drive(agent, 20)
    assert len(calls) == 4  # 2 updates x levels 1, 2
    log = agent.round_log[0]
    assert log["alpha_cap"] == pytest.approx(0.787, abs=1e-3) and log["updates"] == 1
    assert not np.isnan(log["q_spread"]) and 0 <= log["argmax_changed"] <= 1
    assert {f"acc_l{m}" for m in range(3)} <= set(log)
    uni = tiny_agent("uniform_mix")
    drive(uni, 20)
    assert np.allclose(uni.mixer.alpha, 1 / 3)


def test_cond_input_modes():
    for mode in ("mix", "top_level", "sample_level"):
        conds = drive(tiny_agent(cond_input=mode), 25)
        assert np.allclose(conds.sum(1), 1, atol=1e-5)
    with pytest.raises(ValueError):
        tiny_agent(cond_input="median")


def test_agent_is_deterministic():
    a, b = tiny_agent(), tiny_agent()
    ca, cb = drive(a, 30), drive(b, 30)
    assert np.array_equal(ca, cb)
    assert a.round_log[-1]["q_spread"] == b.round_log[-1]["q_spread"]


CKPTS = ("models/npc/predicted_seed0.pt", "models/predictor.pt", "models/env_model.pt")


@pytest.mark.skipif(not all(Path(p).exists() for p in CKPTS), reason="needs the trained checkpoints")
def test_mbom_grid_smoke_writes_trace_and_is_deterministic(tmp_path):
    from scripts.eval_mbom import prepare
    cfg = prepare(load_config("configs/experiments/mbom_smoke.yaml",
                              ["eval.rounds=2", "eval.seeds=[0]", "eval.opponents=[rusher]", "mbom.update_every=15"]))
    players = load_config(cfg["players_config"])
    outs = []
    for run in ("a", "b"):
        rows, _ = run_grid(cfg, ENV, players, tmp_path / run, ["mbom", "ppo_predicted"])
        cell = tmp_path / run / "mbom" / "rusher" / "seed_0"
        assert (cell / "metrics.csv").exists() and (cell / "mbom_rounds.csv").exists()
        with open(cell / "mbom_rounds.csv", encoding="utf-8") as f:
            trace = [{k: v for k, v in r.items() if k != "update_seconds"} for r in csv.DictReader(f)]  # wall clock
        outs.append((trace, (cell / "rounds.csv").read_text()))
        assert "acc_mixed" in rows[0] and "acc_mixed" not in rows[1]
        assert len(trace) == 2 and sum(int(r["updates"]) for r in trace) >= 1
    assert outs[0] == outs[1]

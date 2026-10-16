import numpy as np
import torch

from adaptive_npc.config import load_config
from adaptive_npc.env.observations import OBS_DIM
from adaptive_npc.eval.predictor_eval import archetype_ceiling, mixer_dist, per_archetype_rows, recovery_rows
from adaptive_npc.models.player_model import (FEAT_DIM, OnlinePredictor, WindowData, build_model)
from adaptive_npc.models.predictor_data import collect_match

ENV = load_config("configs/env.yaml")
PLAYERS = load_config("configs/players.yaml")
CFG = load_config("configs/experiments/predictor_smoke.yaml")


def test_collect_deterministic_and_round_flags():
    a = collect_match(ENV, PLAYERS, "scripted_npc", "rusher", 3, seed=5)
    b = collect_match(ENV, PLAYERS, "scripted_npc", "rusher", 3, seed=5)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a["round_start"].sum() == 3 and a["obs"].shape[1] == OBS_DIM


def test_windows_do_not_cross_rounds_and_match_online():
    d = collect_match(ENV, PLAYERS, "random", "mixer", 3, seed=1)
    H = 8
    w = WindowData(d["obs"], d["a_npc"], d["a_pl"], d["round_start"], H)
    starts = np.where(d["round_start"])[0]
    s = int(starts[1])                       # first step of round 2: only the last window row is real
    x, y = w.batch(np.array([s, s + 3]))
    assert (x[0, :-1] == 0).all() and x[0, -1].abs().sum() > 0
    assert (x[1, :H - 4] == 0).all()         # 4 real rows (steps s..s+3)
    # the stateful online predictor builds the identical window
    model = build_model({"hidden": 16, "layers": 1})
    op = OnlinePredictor(model, H)
    for i in range(s, s + 4):
        if d["round_start"][i]:
            op.reset_round()
        op.predict(d["obs"][i])
        win = op._pending
        if i < s + 3:
            op.observe(d["a_npc"][i], d["a_pl"][i])
    assert torch.allclose(win, x[1])
    assert x.shape[-1] == FEAT_DIM


def test_online_finetune_changes_weights_frozen_does_not():
    d = collect_match(ENV, PLAYERS, "random", "rusher", 2, seed=2)
    model = build_model({"hidden": 16, "layers": 1})
    before = [p.clone() for p in model.parameters()]
    for ft in (None, CFG["online"]):
        op = OnlinePredictor(model, 8, ft)
        for i in range(len(d["obs"])):
            if d["round_start"][i]:
                op.reset_round()
            op.predict(d["obs"][i])
            op.observe(d["a_npc"][i], d["a_pl"][i])
        changed = any(not torch.equal(a, b) for a, b in zip(before, op.model.parameters()))
        assert changed == (ft is not None)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))  # original untouched


def test_mixer_ceiling_is_max_weight():
    acc, ent, _ = archetype_ceiling("mixer", PLAYERS)
    p = mixer_dist(PLAYERS)
    assert acc == p.max() == 0.25
    assert abs(ent + (p * np.log(p)).sum()) < 1e-9


def test_eval_smoke_runs():
    model = build_model(CFG["model"]).eval()
    rows = per_archetype_rows(ENV, PLAYERS, CFG, model)
    assert {r["model"] for r in rows} == {"gru_frozen", "gru_online", "majority", "repeat_last"}
    assert all(0 <= r["top1_acc"] <= 1 for r in rows)
    rec = recovery_rows(ENV, PLAYERS, CFG, model)
    assert any(r["target"] == "all" and r["bin_start"] == -1 for r in rec)

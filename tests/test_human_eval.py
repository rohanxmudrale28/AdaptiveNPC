"""Human evaluation (M13): plan / counterbalancing, session logging with a fake play, analysis on synthetic logs."""
import json

import numpy as np
import pytest

from adaptive_npc.config import load_config
from scripts import analyze_human as AH
from scripts import human_eval as HE

CFG = load_config("configs/human_eval.yaml")


def test_counterbalanced_order_and_plan():
    orders = [HE.npc_order(f"P{k:02d}", CFG["npcs"]) for k in range(1, 5)]
    assert orders[0] == orders[2] and orders[1] == orders[3] and orders[0] != orders[1]
    assert HE.npc_order("alice_x", CFG["npcs"]) in (CFG["npcs"], CFG["npcs"][::-1])
    plan = HE.build_plan("P01", CFG)
    assert [m["phase"] for m in plan] == ["practice", "test", "test", "test", "test"]
    assert plan[0]["opponent"] == "dummy"
    assert [m["label"] for m in plan[1:]] == ["A", "A", "B", "B"]
    assert [m["opponent"] for m in plan[1:]] == ["scripted"] * 2 + ["ppo_predicted"] * 2
    assert all(m["rounds"] == 10 for m in plan[1:])
    assert [m["opponent"] for m in HE.build_plan("P02", CFG)[1:]] == ["ppo_predicted"] * 2 + ["scripted"] * 2


def test_rating_prompt():
    assert HE.ask_rating(lambda _: "4") == 4
    assert HE.ask_rating(lambda _: "") is None
    answers = iter(["9", "2"])
    assert HE.ask_rating(lambda _: next(answers)) == 2
    answers = iter(["x", "y"])
    assert HE.ask_rating(lambda _: next(answers)) is None


def fake_summary(rounds, npc_method, p_win, rng, completed=True, predicts=False):
    rs = [{"round": i, "winner": "human" if rng.random() < p_win else "npc", "hp_npc": 0, "hp_human": 0, "steps": 10,
           "damage_dealt": 0, "damage_taken": 0, "pred_accuracy": (0.5 if predicts else None)} for i in range(rounds)]
    return {"npc_method": npc_method, "completed": completed, "rounds": rs, "path": "x"}


def test_session_runs_and_logs_with_fake_play(tmp_path):
    rng = np.random.default_rng(0)
    calls = []

    def fake_play(argv):
        calls.append(argv)
        m = argv[argv.index("--opponent") + 1]
        return fake_summary(int(argv[argv.index("--rounds") + 1]), m, 0.5, rng, predicts=m == "ppo_predicted")

    s = HE.main(["--participant", "P03", "--out-dir", str(tmp_path)], play_main=fake_play, input_fn=lambda _: "3")
    assert len(calls) == 5 and all("--blind" in c for c in calls)
    assert calls[0][calls[0].index("--opponent") + 1] == "dummy"
    assert "--participant" in calls[1] and "P03" in calls[1]
    saved = json.loads((tmp_path / "P03_session.json").read_text())
    assert saved["finished"] and len(saved["matches"]) == 5
    assert "difficulty_rating" not in saved["matches"][0] and saved["matches"][1]["difficulty_rating"] == 3
    assert set(saved) == {"participant", "tick_rate", "order", "config", "started_utc", "matches", "finished"}


def test_incomplete_match_stops_session(tmp_path):
    def fake_play(argv):
        return None

    s = HE.main(["--participant", "P04", "--out-dir", str(tmp_path), "--no-rating"], play_main=fake_play, input_fn=lambda _: "")
    assert len(s["matches"]) == 1 and not s["matches"][0]["completed"]


def test_bad_participant_id_and_dry_run(capsys, tmp_path):
    with pytest.raises(SystemExit):
        HE.main(["--participant", "Jane Doe"])
    assert HE.main(["--participant", "P01", "--dry-run"]) is None
    assert "Opponent A" in capsys.readouterr().out


def write_synthetic(directory, n_participants=6, seed=0):
    """Humans improve vs scripted (0.3 -> 0.8) but not vs ppo_predicted (flat 0.3)."""
    rng = np.random.default_rng(seed)
    for k in range(1, n_participants + 1):
        order = HE.npc_order(f"P{k:02d}", CFG["npcs"])
        matches = [{"phase": "practice", "label": "PRACTICE", "opponent": "dummy", "completed": True,
                    "summary": fake_summary(3, "camper", 0.9, rng)}]
        for npc in order:
            for rep in range(2):
                p = (0.3 + 0.5 * rep) if npc == "scripted" else 0.3
                matches.append({"phase": "test", "label": "A", "opponent": npc, "completed": True, "difficulty_rating": 3,
                                "summary": fake_summary(10, npc, p, rng, predicts=npc == "ppo_predicted")})
        (directory / f"P{k:02d}_session.json").write_text(json.dumps(
            {"participant": f"P{k:02d}", "order": {"A": order[0], "B": order[1]}, "matches": matches}))


def test_analysis_on_synthetic_logs(tmp_path):
    write_synthetic(tmp_path, 8)
    # an incomplete match and practice must be ignored
    s = AH.load_sessions(tmp_path)
    assert len(s) == 8 and all(set(p["npcs"]) == {"scripted", "ppo_predicted"} for p in s)
    rows, text = AH.analyze(s)
    assert "n is small" in text and "exploratory" in text
    by = {(r["metric"]): r for r in rows if r["metric"] != "participant"}
    gain_s = by["win-rate gain, match 2 - match 1, vs scripted"]
    gain_p = by["win-rate gain, match 2 - match 1, vs ppo_predicted"]
    assert gain_s["mean"] > 0.2 and gain_s["mean"] > gain_p["mean"]
    paired = by["win-rate gain: scripted - ppo_predicted"]
    assert paired["n"] == 8 and paired["ci_lo"] < paired["mean"] < paired["ci_hi"]
    assert by["prediction accuracy of ppo_predicted"]["mean"] == pytest.approx(0.5)
    assert by["prediction accuracy of scripted"]["n"] == 0
    curve, cnt = AH.per_round_curve(s, "scripted")
    assert len(curve) == 20 and curve[10:].mean() > curve[:10].mean() and (cnt == 8).all()


def test_analysis_cli_writes_outputs_and_handles_tiny_n(tmp_path, capsys):
    write_synthetic(tmp_path, 1)
    AH.main(["--dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    out = tmp_path / "out"
    for f in ("human_summary.csv", "human_curve.csv", "human_summary.txt", "human_eval.png"):
        assert (out / f).stat().st_size > 0
    assert "n is small" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        AH.main(["--dir", str(tmp_path / "empty")])

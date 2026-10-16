import pytest

from adaptive_npc.config import apply_overrides, load_config


def test_load_and_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 0\nppo:\n  lr: 0.001\n  n_envs: 8\n", encoding="utf-8")
    cfg = load_config(p, ["ppo.lr=3e-4", "ppo.n_envs=4", "new.key=[1,2]", "flag=true"])
    assert cfg["ppo"]["lr"] == pytest.approx(3e-4)
    assert cfg["ppo"]["n_envs"] == 4
    assert cfg["new"]["key"] == [1, 2]
    assert cfg["flag"] is True
    assert cfg["seed"] == 0


def test_override_does_not_mutate_input():
    base = {"a": {"b": 1}}
    out = apply_overrides(base, ["a.b=2"])
    assert base["a"]["b"] == 1 and out["a"]["b"] == 2


def test_bad_override():
    with pytest.raises(ValueError):
        apply_overrides({}, ["nokey"])

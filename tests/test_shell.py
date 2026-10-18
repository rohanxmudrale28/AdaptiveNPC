"""Game shell: UI state machine vs game state, NPC adapters (smoke per option, missing checkpoints), retro view, play flags."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pathlib import Path

import numpy as np
import pygame
import pytest

from adaptive_npc.config import apply_overrides, load_config
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.npc.live import MissingCheckpointError, build_live_npc, required_checkpoints
from adaptive_npc.render.retro import RetroView
from adaptive_npc.render.sprites import SIZE, SpriteSet
from adaptive_npc.ui import shell as S
from adaptive_npc.ui.session import MatchSession

TICK = 4.0
DT = 1.0 / TICK  # exactly representable: one game step per update


@pytest.fixture(scope="module")
def cfgs():
    ui = load_config("configs/ui.yaml")
    ui["tags"]["CUSTOM"] = "silver"
    return {"ui": ui, "render": load_config("configs/render.yaml"), "env": load_config("configs/env.yaml"),
            "players": load_config("configs/players.yaml")}


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


def snapshot(g):
    return (list(g.x), list(g.y), list(g.face), list(g.hp), list(g.sta), list(g.cd), list(g.stun), g.t, g.done,
            g.winner, repr(g.rng.bit_generator.state))


def make_shell(cfgs, rounds=2, env_over=None, **kw):
    env = apply_overrides(cfgs["env"], env_over or [])
    ended = []

    def factory(enemy, rate):
        npc = build_live_npc(enemy["method"], cfgs["ui"], env, cfgs["players"], 0)
        return MatchSession(env, npc, rounds, 0, "", rate)

    sh = S.Shell(cfgs["ui"], cfgs["ui"]["enemies"], TICK, factory, on_match_end=ended.append, **kw)
    return sh, ended, env


def idle():
    return 0


def go_to_match(sh):
    sh.handle("confirm")           # PRESS START -> menu
    sh.handle("confirm")           # PLAY
    assert sh.state == S.SELECT
    sh.handle("confirm")           # first enemy (training dummy)
    assert sh.state == S.COUNTDOWN


def run_countdown(sh):
    while sh.state == S.COUNTDOWN:
        sh.update(0.1, idle)
    assert sh.state == S.PLAY


# ---------------------------------------------------------------------------------------------- state machine
def test_menus_countdown_and_pause_never_touch_game_state(cfgs):
    sh, _, env = make_shell(cfgs)
    for cmd in ("confirm", "down", "confirm", "back", "up", "confirm", "down", "up", "left", "right", "back",
                "confirm", "confirm"):
        sh.handle(cmd)
        sh.update(0.5, idle)
    # title/controls/select did not start anything; now in the match countdown on a freshly reset game
    assert sh.state == S.COUNTDOWN
    fresh = ArenaGame(env, rng=np.random.default_rng(0))
    fresh.reset()                  # the session resets once more at round start, like the old play loop
    assert snapshot(sh.session.game) == snapshot(fresh)
    before = snapshot(sh.session.game)
    for _ in range(40):
        sh.update(0.1, lambda: 5)  # countdown ignores the human's input
        if sh.state == S.PLAY:
            break
        assert snapshot(sh.session.game) == before
    assert sh.state == S.PLAY
    sh.update(DT, lambda: 3)       # one real step
    assert sh.session.game.t == 1
    sh.handle("back")
    assert sh.state == S.PAUSE
    paused = snapshot(sh.session.game)
    for cmd in ("down", "up", "down", "help", "scanlines", "left", "right"):
        sh.handle(cmd)
        sh.update(5.0, lambda: 5)
        assert snapshot(sh.session.game) == paused
    sh.handle("back")              # resume: the backlog of paused time is dropped
    assert sh.state == S.PLAY
    sh.update(0.01, idle)
    assert snapshot(sh.session.game) == paused


def test_shell_play_equals_direct_stepping(cfgs):
    """The shell adds no steps, rng draws or reordering: same actions -> identical game as stepping directly."""
    sh, _, env = make_shell(cfgs, rounds=1)
    go_to_match(sh)
    run_countdown(sh)
    rng = np.random.default_rng(7)
    actions = [int(a) for a in rng.integers(0, 8, size=30)]
    direct = MatchSession(env, build_live_npc("camper", cfgs["ui"], env, cfgs["players"], 0), 1, 0)
    it = iter(actions)
    for a in actions:
        if sh.state != S.PLAY:
            break
        sh.update(DT, lambda: a)
        direct.tick(a)
        assert snapshot(sh.session.game) == snapshot(direct.game)


def test_round_flow_results_and_match_end_callback(cfgs):
    sh, ended, _ = make_shell(cfgs, rounds=2, env_over=["max_steps=6"])
    go_to_match(sh)
    for rnd in range(2):
        run_countdown(sh)
        while sh.state == S.PLAY:
            sh.update(DT, idle)
        assert sh.state == S.ROUND_END
        sh.handle("confirm")
    assert sh.state == S.RESULTS
    assert len(ended) == 1 and ended[0]["completed"] and ended[0]["rounds_played"] == 2
    assert sum(ended[0]["rounds_won"].values()) == 2
    assert ended[0]["prediction_accuracy"] is None  # the dummy has no predictor
    sh.handle("down")
    sh.handle("confirm")           # MENU
    assert sh.state == S.TITLE and sh.session is None


def test_pause_menu_restart_and_quit_report_partial_matches(cfgs):
    sh, ended, _ = make_shell(cfgs, rounds=3, env_over=["max_steps=4"])
    go_to_match(sh)
    run_countdown(sh)
    while sh.state == S.PLAY:
        sh.update(DT, idle)
    sh.handle("confirm")           # round 1 recorded, countdown of round 2
    sh.handle("back")              # pause
    sh.handle("down")              # RESTART MATCH
    sh.handle("confirm")
    assert len(ended) == 1 and not ended[0]["completed"] and ended[0]["rounds_played"] == 1
    assert sh.state == S.COUNTDOWN and sh.session.round == 0 and sh.session.score == [0, 0, 0]
    sh.handle("back")
    sh.handle("down")
    sh.handle("down")              # QUIT TO MENU with no round played yet: nothing to report
    sh.handle("confirm")
    assert len(ended) == 1 and sh.state == S.TITLE


def test_blind_hides_names_and_skips_select(cfgs):
    sh, _, _ = make_shell(cfgs, blind=True, label="B", select_enabled=False, enemy_index=4)
    assert sh.opponent_label == "OPPONENT B" and "ECHO" not in sh.opponent_label
    sh.handle("confirm")
    sh.handle("confirm")           # PLAY goes straight to the match
    assert sh.state == S.COUNTDOWN


def test_no_menu_pause_quits(cfgs):
    sh, _, _ = make_shell(cfgs, has_menu=False)
    sh.begin()
    assert sh.state == S.COUNTDOWN
    sh.handle("back")
    assert sh.menu_items[-1] == "QUIT"
    sh.handle("up")
    sh.handle("confirm")
    assert sh.state == S.QUIT


def test_legend_shows_then_hides_and_h_toggles(cfgs):
    sh, _, _ = make_shell(cfgs)
    go_to_match(sh)
    assert sh.legend_visible
    sh.update(cfgs["ui"]["legend_seconds"] + 0.1, idle)
    assert not sh.legend_visible
    sh.handle("help")
    assert sh.legend_visible
    sh.update(60.0, idle)
    assert sh.legend_visible       # pinned by the player
    sh.handle("help")
    assert not sh.legend_visible


def test_speed_option_defaults_to_tick_rate(cfgs):
    sh, _, _ = make_shell(cfgs)
    assert sh.tick_rate == 4.0
    sh.handle("confirm"), sh.handle("confirm")
    sh.handle("right")
    assert sh.tick_rate == 6.0


# ---------------------------------------------------------------------------------------------- NPC options
ENEMY_METHODS = [e["method"] for e in load_config("configs/ui.yaml")["enemies"]]


def test_enemies_cover_every_npc_version(cfgs):
    assert ENEMY_METHODS == ["camper", "scripted", "ppo_none", "ppo_predicted", "mbom"]
    assert set(e["accessory"] for e in cfgs["ui"]["enemies"]) <= {"none", "scarf", "visor", "antennae", "crest"}


@pytest.mark.parametrize("method", ENEMY_METHODS)
def test_each_npc_option_plays_a_round(cfgs, method):
    missing = [p for p, _ in required_checkpoints(method, load_config(cfgs["ui"]["npc_config"]),
                                                  cfgs["ui"]["npc_seed"]) if not Path(p).is_file()]
    if missing:
        pytest.skip(f"checkpoints not available here: {missing}")
    env = apply_overrides(cfgs["env"], ["max_steps=55", "hp=1000"])  # full 55 steps; MBOM updates every 50
    npc = build_live_npc(method, cfgs["ui"], env, cfgs["players"], 0)
    s = MatchSession(env, npc, 1, 0)
    rng = np.random.default_rng(1)
    while not s.tick(int(rng.integers(0, 8))):
        pass
    assert s.finished is False and len(s.rounds) == 1
    s.next_round()
    assert s.finished
    summ = s.summary(True)
    assert summ["npc_method"] == npc.method and summ["rounds_played"] == 1
    if method in ("ppo_predicted", "mbom"):
        assert summ["prediction_steps"] == 55 and 0.0 <= summ["prediction_accuracy"] <= 1.0
    else:
        assert summ["prediction_accuracy"] is None


@pytest.mark.parametrize("method", ["ppo_none", "ppo_predicted", "mbom"])
def test_missing_checkpoint_gives_clear_error(cfgs, method):
    ui = apply_overrides(cfgs["ui"], ["npc_seed=987"])
    with pytest.raises(MissingCheckpointError) as ei:
        build_live_npc(method, ui, cfgs["env"], cfgs["players"], 0)
    msg = str(ei.value)
    assert "seed987.pt" in msg and "scripts.train_npc" in msg and method in msg
    assert isinstance(ei.value, FileNotFoundError)
    assert any("seed987.pt" in p for p, _ in ei.value.missing)


def test_missing_checkpoint_shows_error_on_select_screen_without_leaving_it(cfgs):
    ui = apply_overrides(cfgs["ui"], ["npc_seed=987"])
    shell_cfgs = {**cfgs, "ui": ui}
    sh, _, _ = make_shell(shell_cfgs)
    sh.handle("confirm"), sh.handle("confirm")
    sh.enemy_idx = 2               # ppo_none
    sh.handle("confirm")
    assert sh.state == S.SELECT and sh.session is None
    assert sh.error[0] == "MISSING MODEL FILE" and "seed987.pt" in sh.error[1]
    sh.handle("up")                # moving the cursor clears it
    assert sh.error is None


# ---------------------------------------------------------------------------------------------- view
def test_sprites_build_for_every_look_and_have_all_frames(cfgs):
    ui = cfgs["ui"]
    sets = [SpriteSet("knight", ui["palette"], ui["looks"]["player"])]
    sets += [SpriteSet("robot", ui["palette"], ui["looks"][e["look"]], e["accessory"]) for e in ui["enemies"]]
    for sp in sets:
        for facing in range(4):
            assert set(sp.frames[facing]) == {"idle0", "idle1", "walk0", "walk1", "attack", "block", "ko"}
            assert all(f.get_size() == (SIZE, SIZE) for f in sp.frames[facing].values())
    assert len(ui["palette"]) <= 16


def test_right_facing_is_mirror_of_left(cfgs):
    ui = cfgs["ui"]
    sp = SpriteSet("knight", ui["palette"], ui["looks"]["player"])
    left, right = sp.frames[2]["idle0"], sp.frames[3]["idle0"]
    assert all(left.get_at((x, y)) == right.get_at((SIZE - 1 - x, y)) for x in range(SIZE) for y in range(SIZE))


def drive_view(cfgs, view, sh, frames=3):
    for _ in range(frames):
        sh.update(0.016, idle)
        view.draw(sh, 0.016)


def test_view_renders_every_screen_without_mutating_game(cfgs):
    sh, _, env = make_shell(cfgs, rounds=1)
    view = RetroView(cfgs["ui"], cfgs["render"], env["arena_size"])
    assert view.canvas.get_size() == (320, 240) and view.window.get_size() == (960, 720)
    for cmd in ("confirm", "confirm"):
        drive_view(cfgs, view, sh)
        sh.handle(cmd)
    drive_view(cfgs, view, sh)     # select
    sh.handle("confirm")
    drive_view(cfgs, view, sh)     # countdown
    run_countdown(sh)
    before = None
    for a in (5, 3, 6, 7, 1, 2, 5, 5, 0):
        sh.update(DT, lambda a=a: a)
        view.draw(sh, DT)
    sh.scanlines = True
    sh.handle("back")
    before = snapshot(sh.session.game)
    drive_view(cfgs, view, sh, 5)  # pause overlay
    assert snapshot(sh.session.game) == before
    sh.handle("back")
    while sh.state == S.PLAY:
        sh.update(DT, idle)
        view.draw(sh, DT)
    drive_view(cfgs, view, sh)     # round end
    sh.handle("confirm")
    drive_view(cfgs, view, sh)     # results
    assert sh.state == S.RESULTS


def test_title_select_controls_use_only_palette_colours(cfgs):
    ui = cfgs["ui"]
    sh, _, env = make_shell(cfgs)
    view = RetroView(ui, cfgs["render"], env["arena_size"])
    pal = {tuple(v) for v in ui["palette"].values()}
    seen = set()
    for step in ("title", "menu", "select", "controls"):
        if step == "menu":
            sh.handle("confirm")
        elif step == "select":
            sh.handle("confirm")
        elif step == "controls":
            sh.handle("back"), sh.handle("back")  # title menu
            sh.handle("down"), sh.handle("confirm")
        sh.update(0.0, idle)
        view.draw(sh, 0.3)
        for x in range(view.W):
            for y in range(view.H):
                seen.add(tuple(view.canvas.get_at((x, y)))[:3])
    assert seen <= pal


def test_scanlines_toggle_changes_window_not_canvas(cfgs):
    sh, _, env = make_shell(cfgs)
    view = RetroView(cfgs["ui"], cfgs["render"], env["arena_size"])
    sh.update(0.0, idle)
    view.draw(sh, 0.0)
    a = pygame.image.tobytes(view.window, "RGB")
    sh.handle("scanlines")
    view.draw(sh, 0.0)
    b = pygame.image.tobytes(view.window, "RGB")
    assert a != b and sh.scanlines


# ---------------------------------------------------------------------------------------------- play.py flags
def test_play_flag_validation_fails_early():
    from scripts import play
    with pytest.raises(SystemExit):
        play.main(["--blind"])                                            # needs --opponent
    with pytest.raises(SystemExit):
        play.main(["--blind", "--opponent", "rusher", "--participant", "Jane Doe"])


def test_find_enemy_by_key_method_alias_and_custom(cfgs):
    from scripts import play
    en = cfgs["ui"]["enemies"]
    assert play.find_enemy(en, "mbom")[0] == 4 and play.find_enemy(en, "seer")[0] == 3
    assert play.find_enemy(en, "scripted_npc")[0] == 1
    idx, lst = play.find_enemy(en, "rusher")
    assert idx == len(en) and lst[idx]["method"] == "rusher" and len(lst) == len(en) + 1

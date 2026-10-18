"""Demo panel: pixel baseline with the panel off, layout, read-only behaviour, async MBOM update."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import copy
import time
from pathlib import Path

import numpy as np
import pygame
import pytest

from adaptive_npc.config import apply_overrides, load_config
from adaptive_npc.npc.live import build_live_npc, required_checkpoints
from adaptive_npc.render.retro import PANEL_H, PANEL_W, PANEL_X, RetroView
from adaptive_npc.ui import shell as S
from adaptive_npc.ui.session import MatchSession
from tests.frame_hash import fixed_frames
from tests.test_shell import DT, go_to_match, idle, make_shell, run_countdown, snapshot

# sha256 of canvas+window recorded from commit 4a938a0 (before the panel existed), panel off.
BASELINE = {"title": "1fedcae914822e9bbf8249b24e68e2713de9e218eaea1c027745b450dbb6706e",
            "select": "6bd967b95775e01412fc759249d1848d7cf1faded1576612bbaa10286a078a80",
            "match": "7a42b4393f17d67b7514dd453a82145334aaf5530fade80fad69c7ef0a808b54"}


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


def test_panel_off_frames_identical_to_baseline(cfgs):
    # ECHO's tag was deliberately changed EXPERT -> HARD after the baseline; restore it so the select screen
    # is compared like for like. Everything else (arena, HUD, sprites, menus, palette) is untouched.
    old = copy.deepcopy(cfgs["ui"])
    old["enemies"][4]["tag"] = "EXPERT"
    assert fixed_frames({**cfgs, "ui": old}, make_shell) == BASELINE


def match_frame(cfgs, panel):
    """Fixed mid-fight state with real ticks (so the panel has data); returns (canvas bytes, view, shell)."""
    sh, _, env = make_shell({**cfgs, "env": apply_overrides(cfgs["env"], ["random_start=false"])}, panel=panel)
    view = RetroView(cfgs["ui"], cfgs["render"], env["arena_size"])
    sh.handle("confirm"), sh.handle("confirm"), sh.handle("confirm")
    run_countdown(sh)
    sh.legend_visible, sh.legend_timer = False, None
    g = sh.session.game
    view.sync(g)
    for a1 in (3, 5, 6):
        sh.session.tick(a1)
        view.time += 0.3
        view.sync(g)
    view.time = 0.37
    view.draw(sh, 0.0)
    return pygame.image.tobytes(view.canvas, "RGB"), view, sh


def test_panel_only_touches_the_margin_right_of_the_arena(cfgs):
    off, view, _ = match_frame(cfgs, False)
    on, _, _ = match_frame(cfgs, True)
    assert off != on
    w = view.W
    changed = {(i // 3 % w, i // 3 // w) for i in range(len(off)) if off[i] != on[i]}
    xs, ys = [c[0] for c in changed], [c[1] for c in changed]
    ax1 = view.ax + view.n * 16 + 2  # arena incl. its wall frame
    assert min(xs) >= PANEL_X and max(xs) < PANEL_X + PANEL_W and ax1 <= PANEL_X
    assert min(ys) >= view.ay - 2 and max(ys) < view.ay - 2 + PANEL_H
    assert (view.ax, view.ay) == (88, 52) and view.W == 320 and view.window.get_size() == (960, 720)
    assert view.ay - 2 + PANEL_H <= view.ay + view.n * 16 + 8  # stays above the bottom text strip


def test_panel_does_not_change_game_state(cfgs):
    runs = []
    for panel in (False, True):
        sh, _, env = make_shell({**cfgs, "env": apply_overrides(cfgs["env"], ["random_start=false"])}, panel=panel)
        view = RetroView(cfgs["ui"], cfgs["render"], env["arena_size"])
        go_to_match(sh)
        run_countdown(sh)
        snaps = []
        rng = np.random.default_rng(3)
        for _ in range(25):
            a = int(rng.integers(0, 8))
            sh.update(DT, lambda a=a: a)
            view.draw(sh, DT)
            if panel:
                sh.handle("panel"), view.draw(sh, 0.0), sh.handle("panel")
            snaps.append(snapshot(sh.session.game))
            if sh.state != S.PLAY:
                break
        runs.append(snaps)
    assert runs[0] == runs[1]


def test_blind_never_shows_panel(cfgs):
    sh, _, _ = make_shell(cfgs, blind=True, select_enabled=False, panel=True)
    assert not sh.panel
    sh.handle("panel")
    assert not sh.panel


def test_panel_toggle_and_start_state(cfgs):
    sh, _, _ = make_shell(cfgs)
    assert not sh.panel
    sh.handle("panel")
    assert sh.panel
    sh.handle("panel")
    assert not sh.panel
    assert make_shell(cfgs, panel=True)[0].panel


def test_session_records_prediction_and_running_accuracy(cfgs):
    class FakeNPC:
        method, predicts, pred, alpha, alpha_cap = "fake", True, None, None, None

        def reset_round(self):
            pass

        def end_round(self):
            pass

        def observe(self, *a):
            pass

        def act(self, g):
            self.pred = np.eye(8)[5]  # always guesses ATTACK
            return 0

    s = MatchSession(apply_overrides(cfgs["env"], ["hp=1000"]), FakeNPC(), 1, 0)
    assert s.running_accuracy is None and s.last_actual is None
    for a in (5, 5, 3, 5):
        s.tick(a)
    assert s.last_actual == 5 and s.last_pred.argmax() == 5
    assert s.running_accuracy == pytest.approx(0.75)


def test_screenshot_is_saved(cfgs, tmp_path):
    _, view, _ = match_frame(cfgs, True)
    path = view.save_screenshot(tmp_path)
    assert path.exists() and pygame.image.load(str(path)).get_size() == (960, 720)


def test_alpha_only_for_mbom(cfgs):
    for method in ("camper", "scripted"):
        assert build_live_npc(method, cfgs["ui"], cfgs["env"], cfgs["players"], 0).alpha is None


# ---------------------------------------------------------------------------------------------- async MBOM
def _mbom_ui_or_skip(cfgs, **over):
    ui = apply_overrides(cfgs["ui"], [f"{k}={v}" for k, v in over.items()])
    missing = [p for p, _ in required_checkpoints("mbom", load_config(ui["npc_config"]), ui["npc_seed"])
               if not Path(p).is_file()]
    if missing:
        pytest.skip(f"checkpoints not available here: {missing}")
    return ui


def test_mbom_update_runs_off_thread_and_loop_keeps_ticking(cfgs):
    ui = _mbom_ui_or_skip(cfgs)
    env = apply_overrides(cfgs["env"], ["hp=1000", "max_steps=400"])
    npc = build_live_npc("mbom", ui, env, cfgs["players"], 0)
    agent = npc.agent
    assert hasattr(agent, "poll")
    real_compute = agent.compute_update
    started = []

    def slow_compute(iops, wins, raws, acts, first):
        started.append(time.perf_counter())
        time.sleep(1.5)  # stands in for the ~3 s imagination update
        return real_compute(iops, wins, raws, acts, first)

    agent.compute_update = slow_compute
    s = MatchSession(env, npc, 1, 0)
    rng = np.random.default_rng(0)
    durations = []
    while agent.n_updates == 0 and len(durations) < 120:
        t0 = time.perf_counter()
        s.tick(int(rng.integers(0, 8)))
        durations.append(time.perf_counter() - t0)
        if started:
            time.sleep(0.02)
    assert started, "update never triggered"
    assert agent.busy or agent.n_updates == 1
    assert len(durations) > 52  # kept ticking after the trigger at step 50
    assert max(durations) < 0.5, f"a tick stalled for {max(durations):.2f}s while the update ran"
    deadline = time.time() + 120
    while agent.n_updates == 0 and time.time() < deadline:
        time.sleep(0.05)
        agent.poll()
    assert agent.n_updates == 1 and not agent.busy


def test_countdown_and_shell_not_stalled_by_update(cfgs):
    ui = _mbom_ui_or_skip(cfgs)
    holder = {}

    def factory(enemy, rate):
        env = apply_overrides(cfgs["env"], ["hp=1000"])
        npc = build_live_npc("mbom", ui, env, cfgs["players"], 0)
        real = npc.agent.compute_update

        def slow(*a):
            time.sleep(1.0)
            return real(*a)

        npc.agent.compute_update = slow
        holder["npc"] = npc
        return MatchSession(env, npc, 1, 0, "", rate)

    sh = S.Shell(ui, ui["enemies"], 8.0, factory)
    sh.handle("confirm"), sh.handle("confirm")
    sh.enemy_idx = 4
    sh.handle("confirm")
    run_countdown(sh)
    worst = 0.0
    for _ in range(70):
        t0 = time.perf_counter()
        sh.update(1 / 8, idle)
        worst = max(worst, time.perf_counter() - t0)
    assert holder["npc"].agent.busy or holder["npc"].agent.n_updates
    assert worst < 0.5
    sh.handle("back"), sh.handle("down"), sh.handle("confirm")  # restart: must not wait for the old worker
    t0 = time.perf_counter()
    while sh.state == S.COUNTDOWN:
        sh.update(0.1, idle)
    assert sh.state == S.PLAY and time.perf_counter() - t0 < 1.0


def test_synchronous_mode_is_plain_agent(cfgs):
    ui = _mbom_ui_or_skip(cfgs, mbom_async="false")
    npc = build_live_npc("mbom", ui, cfgs["env"], cfgs["players"], 0)
    assert not hasattr(npc.agent, "poll")

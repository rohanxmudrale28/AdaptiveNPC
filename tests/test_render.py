import os
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from adaptive_npc.config import load_config
from adaptive_npc.env.actions import Action
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.render.clock import TickClock
from adaptive_npc.render.input import KeyboardInput, scheme_keys
from adaptive_npc.render.view import ArenaView


@pytest.fixture(scope="module")
def cfgs():
    return load_config("configs/render.yaml"), load_config("configs/env.yaml")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


class Pressed(dict):
    def __missing__(self, key):
        return False


def test_tick_clock_rate_independent_of_fps():
    for tick_rate in (8, 20):
        for fps in (30, 60, 144):
            clock, steps = TickClock(tick_rate), 0
            for _ in range(fps * 10):  # 10 simulated seconds
                steps += clock.advance(1.0 / fps)
            assert abs(steps - tick_rate * 10) <= 1


def test_tick_clock_drops_backlog():
    clock = TickClock(8, max_catchup=4)
    assert clock.advance(5.0) == 4
    assert clock.advance(0.0) == 0


def test_tick_clock_rejects_bad_rate():
    with pytest.raises(ValueError):
        TickClock(0)


def ev(type_, **kw):
    return pygame.event.Event(type_, **kw)


def mouse_input(cfgs, now=None):
    t = [100.0] if now is None else now
    return KeyboardInput(cfgs[0]["keys"], clock=lambda: t[0]), t


def test_default_bindings_are_mouse_and_q(cfgs):
    keys = cfgs[0]["keys"]
    assert (keys["attack"], keys["block"], keys["dodge"]) == (["mouse_left"], ["mouse_right"], ["q"])
    kb, _ = mouse_input(cfgs)
    for dead in (pygame.K_j, pygame.K_k, pygame.K_l, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_LCTRL):
        kb.handle_event(ev(pygame.KEYDOWN, key=dead))
        assert kb.poll(Pressed({dead: True})) == Action.IDLE


def test_left_click_maps_to_attack(cfgs):
    kb, _ = mouse_input(cfgs)
    kb.handle_event(ev(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0)))
    kb.handle_event(ev(pygame.MOUSEBUTTONUP, button=1, pos=(0, 0)))   # released before the tick: still counts once
    assert kb.poll(Pressed()) == Action.ATTACK
    assert kb.poll(Pressed()) == Action.IDLE


def test_right_click_held_maps_to_block_across_steps(cfgs):
    kb, _ = mouse_input(cfgs)
    kb.handle_event(ev(pygame.MOUSEBUTTONDOWN, button=3, pos=(0, 0)))
    assert [kb.poll(Pressed()) for _ in range(3)] == [Action.BLOCK] * 3
    kb.handle_event(ev(pygame.MOUSEBUTTONUP, button=3, pos=(0, 0)))
    assert kb.poll(Pressed()) == Action.IDLE


def test_q_maps_to_dodge(cfgs):
    kb, _ = mouse_input(cfgs)
    kb.handle_event(ev(pygame.KEYDOWN, key=pygame.K_q))
    assert kb.poll(Pressed()) == Action.DODGE
    assert kb.poll(Pressed({pygame.K_q: True})) == Action.DODGE


def test_clicks_ignored_when_not_active_play(cfgs):
    kb, _ = mouse_input(cfgs)
    for button in (1, 3):
        kb.handle_event(ev(pygame.MOUSEBUTTONDOWN, button=button, pos=(0, 0)), active=False)
    assert kb.poll(Pressed()) == Action.IDLE
    # a button pressed outside play and still down when play starts does not count, and its release is harmless
    kb.handle_event(ev(pygame.MOUSEBUTTONUP, button=3, pos=(0, 0)), active=False)
    assert kb.poll(Pressed()) == Action.IDLE


def test_release_outside_play_clears_held_block(cfgs):
    kb, _ = mouse_input(cfgs)
    kb.handle_event(ev(pygame.MOUSEBUTTONDOWN, button=3, pos=(0, 0)))
    assert kb.poll(Pressed()) == Action.BLOCK
    kb.handle_event(ev(pygame.MOUSEBUTTONUP, button=3, pos=(0, 0)), active=False)   # e.g. released during pause
    assert kb.poll(Pressed()) == Action.IDLE


def test_focus_click_is_not_an_attack(cfgs):
    kb, t = mouse_input(cfgs)
    kb.handle_event(ev(pygame.WINDOWFOCUSGAINED))
    kb.handle_event(ev(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0)))
    assert kb.poll(Pressed()) == Action.IDLE
    t[0] += 1.0
    kb.handle_event(ev(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0)))
    assert kb.poll(Pressed()) == Action.ATTACK


def test_losing_focus_drops_held_block(cfgs):
    kb, _ = mouse_input(cfgs)
    kb.handle_event(ev(pygame.MOUSEBUTTONDOWN, button=3, pos=(0, 0)))
    kb.handle_event(ev(pygame.WINDOWFOCUSLOST))
    assert kb.poll(Pressed()) == Action.IDLE


def test_keyboard_scheme_is_the_old_jkl_bindings(cfgs):
    keys = scheme_keys(cfgs[0], "keyboard")
    assert (keys["attack"], keys["block"], keys["dodge"]) == (["j", "space"], ["k", "left shift"], ["l", "left ctrl"])
    assert keys["up"] == cfgs[0]["keys"]["up"]
    kb = KeyboardInput(keys)
    kb.handle_event(ev(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0)))
    assert kb.poll(Pressed()) == Action.IDLE            # clicks do nothing in keyboard mode
    assert kb.poll(Pressed({pygame.K_k: True})) == Action.BLOCK
    kb.handle_event(ev(pygame.KEYDOWN, key=pygame.K_l))
    assert kb.poll(Pressed()) == Action.DODGE


def test_play_input_flag_and_deprecated_keys_alias(cfgs, capsys):
    from scripts import play
    parse = play.build_parser().parse_args
    assert parse([]).input is None and parse([]).keys is None             # default comes from render.yaml (all)
    assert play.resolve_scheme(parse([]), cfgs[0]) == "all"
    for scheme in ("all", "keyboard", "kbm", "controller"):
        assert play.resolve_scheme(parse(["--input", scheme]), cfgs[0]) == scheme
    capsys.readouterr()
    assert play.resolve_scheme(parse(["--keys", "keyboard"]), cfgs[0]) == "keyboard"
    assert play.resolve_scheme(parse(["--keys", "mouse"]), cfgs[0]) == "kbm"
    assert "deprecated" in capsys.readouterr().err
    assert play.resolve_scheme(parse(["--keys", "mouse", "--input", "controller"]), cfgs[0]) == "controller"
    with pytest.raises(SystemExit):
        parse(["--input", "gamepad"])
    with pytest.raises(SystemExit):
        play.resolve_scheme(parse([]), {**cfgs[0], "input": "bogus"})


def test_legend_lists_the_active_schemes(cfgs):
    from adaptive_npc.render.retro import RetroView
    ui = load_config("configs/ui.yaml")
    arena = cfgs[1]["arena_size"]
    lines = {s: RetroView(ui, {**cfgs[0], "input": s}, arena).legend_lines()
             for s in ("keyboard", "kbm", "controller", "all")}
    assert lines["kbm"][0] == "MOVE WASD   ATTACK LMB   BLOCK RMB   DASH Q"
    assert lines["keyboard"][0] == "MOVE WASD   ATTACK J   BLOCK K   DASH L"
    assert lines["controller"][0] == "MOVE STICK   ATTACK A   BLOCK B   DASH Y"
    assert lines["all"] == ["MOVE WASD/STICK   ATTACK J/LMB/A", "BLOCK K/RMB/B (HOLD)   DASH L/Q/Y",
                            "H LEGEND   F SCANLINES   ESC/START PAUSE"]
    assert lines["kbm"][1] == "H LEGEND   F SCANLINES   ESC PAUSE"
    assert lines["controller"][1].endswith("ESC/START PAUSE")
    assert RetroView(ui, cfgs[0], arena).scheme == "all"                  # render.yaml default


def test_keyboard_held_and_latched(cfgs):
    kb = KeyboardInput(scheme_keys(cfgs[0], "keyboard"))
    assert kb.poll(Pressed()) == Action.IDLE
    assert kb.poll(Pressed({pygame.K_w: True})) == Action.UP
    assert kb.poll(Pressed({pygame.K_k: True})) == Action.BLOCK
    # held attack beats held movement
    assert kb.poll(Pressed({pygame.K_d: True, pygame.K_j: True})) == Action.ATTACK
    # a tap released before the tick is not lost, and is consumed once
    kb.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))
    assert kb.poll(Pressed()) == Action.DODGE
    assert kb.poll(Pressed()) == Action.IDLE


def test_keyboard_control_keys(cfgs):
    kb = KeyboardInput(cfgs[0]["keys"])
    assert kb.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)) == "quit"
    assert kb.handle_event(pygame.event.Event(pygame.QUIT)) == "quit"
    assert kb.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)) == "next_round"


def test_view_draws_without_mutating_state(cfgs):
    render_cfg, env_cfg = cfgs
    g = ArenaGame(env_cfg, rng=np.random.default_rng(0))
    g.step(Action.BLOCK, Action.DODGE)
    before = (list(g.x), list(g.y), list(g.hp), list(g.sta), g.t, list(g.stun))
    view = ArenaView(render_cfg, env_cfg["arena_size"])
    view.draw(g, (1, 2, 3), 0, ("npc", "YOU"))
    view.draw(g, banner=["YOU WIN", "Enter"])
    assert (list(g.x), list(g.y), list(g.hp), list(g.sta), g.t, list(g.stun)) == before
    assert view.screen.get_size() == (9 * 64, 72 + 9 * 64)


def test_core_and_env_do_not_import_pygame():
    code = ("import sys, adaptive_npc.env.core, adaptive_npc.env.gym_env, adaptive_npc.env.match, "
            "adaptive_npc.env.vec_env; sys.exit(int('pygame' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

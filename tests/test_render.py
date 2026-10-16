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
from adaptive_npc.render.input import KeyboardInput
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


def test_keyboard_held_and_latched(cfgs):
    kb = KeyboardInput(cfgs[0]["keys"])
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

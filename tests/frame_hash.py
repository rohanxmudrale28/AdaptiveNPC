"""Fixed-state frame hashes used to pin the look of the game (panel off) across changes."""
import hashlib

import pygame

from adaptive_npc.config import apply_overrides
from adaptive_npc.render.retro import RetroView
from adaptive_npc.ui import shell as S


def _h(view):
    return hashlib.sha256(pygame.image.tobytes(view.canvas, "RGB") + pygame.image.tobytes(view.window, "RGB")).hexdigest()


def fixed_frames(cfgs, make_shell, **view_kw):
    """sha256 of canvas+window for the title, select and a fixed mid-fight frame."""
    sh, _, env = make_shell({**cfgs, "env": apply_overrides(cfgs["env"], ["random_start=false"])})
    view = RetroView(cfgs["ui"], cfgs["render"], env["arena_size"], **view_kw)
    out = {}
    view.time = 0.0
    view.draw(sh, 0.0)
    out["title"] = _h(view)
    sh.handle("confirm"), sh.handle("confirm")
    view.draw(sh, 0.0)
    out["select"] = _h(view)
    sh.handle("confirm")
    while sh.state == S.COUNTDOWN:
        sh.update(0.1, lambda: 0)
    sh.legend_visible, sh.legend_timer = False, None
    g = sh.session.game
    view.sync(g)
    for a0, a1 in ((5, 3), (6, 5), (3, 6)):
        g.step(a0, a1)
        view.time += 0.3
        view.sync(g)
    view.time = 0.37
    view.draw(sh, 0.0)
    out["match"] = _h(view)
    return out

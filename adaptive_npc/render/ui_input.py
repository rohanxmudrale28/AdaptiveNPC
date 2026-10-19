"""pygame events -> shell commands (up/down/left/right/confirm/back/help/scanlines/quit, plus mute / volume_up / volume_down for the sound layer).
With a controller config the stick / D-pad navigate, A confirms and B goes back on menu screens; Start pauses and resumes
a match. During COUNTDOWN / PLAY the face buttons and stick are gameplay only (they never produce menu commands)."""
import time

import pygame

from adaptive_npc.render.controller import AXIS_X, AXIS_Y, DPAD, StickDirection, button_code, norm_axis
from adaptive_npc.ui import shell as S

NAV_BLOCKED = (S.COUNTDOWN, S.PLAY, S.ROUND_END)


class UIKeys:
    def __init__(self, ui_keys, render_keys, controller_cfg=None, clock=time.monotonic):
        self.bind = {}
        for cmd, names in (*((k, ui_keys[k]) for k in ("confirm", "back", "help", "scanlines", "panel", "screenshot",
                                                         "mute", "volume_up", "volume_down")),
                           *((k, render_keys[k]) for k in ("up", "down", "left", "right"))):
            self.bind[cmd] = {pygame.key.key_code(n) for n in names}
        self.pad = None
        self._clock, self._state, self._since = clock, None, 0.0
        if controller_cfg is not None:
            c = controller_cfg
            self.pad = {cmd: {button_code(n) for n in c[cmd]} for cmd in ("confirm", "back", "pause")}
            self.pad_dir = {button_code(b): d for d, b in DPAD.items()}
            self.round_end_delay = float(c["round_end_delay"])
            self.stick = StickDirection(c["deadzone"], c["hysteresis"], c["axis_margin"])
            self._axes = {AXIS_X: 0.0, AXIS_Y: 0.0}

    def observe(self, state):
        """Call once per frame with the shell state (used to decide what controller buttons mean)."""
        if state != self._state:
            self._state, self._since = state, self._clock()

    def command(self, event):
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            for cmd, keys in self.bind.items():
                if event.key in keys:
                    return cmd
        if self.pad is not None:
            return self._pad_command(event)
        return None

    def _pad_command(self, event):
        state = self._state
        if event.type == pygame.CONTROLLERAXISMOTION and event.axis in self._axes:
            self._axes[event.axis] = norm_axis(event.value)
            prev = self.stick.dir
            now = self.stick.update(self._axes[AXIS_X], self._axes[AXIS_Y])   # tracked in play too, so state stays right
            return now if now is not None and now != prev and state not in NAV_BLOCKED else None
        if event.type != pygame.CONTROLLERBUTTONDOWN:
            return None
        b = event.button
        if b in self.pad["pause"]:
            return "back" if state in S.IN_MATCH else "confirm"
        if state in (S.COUNTDOWN, S.PLAY):
            return None
        if b in self.pad_dir:
            return self.pad_dir[b] if state not in NAV_BLOCKED else None
        if b in self.pad["confirm"]:
            if state == S.ROUND_END and self._clock() - self._since < self.round_end_delay:
                return None   # A is also attack: don't skip the round-end screen while still fighting
            return "confirm"
        if b in self.pad["back"]:
            return None if state == S.ROUND_END else "back"   # B is block: only Start pauses from the round-end screen
        return None

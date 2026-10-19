"""Keyboard + mouse -> Action. Taps and clicks are latched until the next game tick; held keys and held mouse
buttons are re-read every tick. Mouse buttons are named in the key config ("mouse_left", "mouse_middle",
"mouse_right"); everything else is a pygame key name. Mouse presses only count while `active` (the PLAY state), and
the click that focuses the window is ignored."""
import time

import pygame

from adaptive_npc.env.actions import Action

_ACTION_KEYS = ("up", "down", "left", "right", "attack", "block", "dodge")
_ACTIONS = {"up": Action.UP, "down": Action.DOWN, "left": Action.LEFT, "right": Action.RIGHT,
            "attack": Action.ATTACK, "block": Action.BLOCK, "dodge": Action.DODGE}
# When several held keys compete: commit actions beat movement.
_HELD_PRIORITY = ("dodge", "attack", "block", "up", "down", "left", "right")
MOUSE_BUTTONS = {"mouse_left": 1, "mouse_middle": 2, "mouse_right": 3}
FOCUS_GRACE = 0.25  # seconds after the window gains focus during which mouse presses are ignored


class KeyboardInput:
    def __init__(self, keys_cfg, clock=time.monotonic):
        self.bind, self.mouse = {}, {}
        for name in list(_ACTION_KEYS) + ["next_round", "quit"]:
            self.bind[name] = [pygame.key.key_code(k) for k in keys_cfg[name] if k not in MOUSE_BUTTONS]
            self.mouse[name] = [MOUSE_BUTTONS[k] for k in keys_cfg[name] if k in MOUSE_BUTTONS]
        self._clock = clock
        self._latched = None
        self._held_mouse = set()
        self._ignore_mouse_until = 0.0

    def _name_of(self, key=None, button=None):
        for name in _ACTION_KEYS:
            if (key is not None and key in self.bind[name]) or (button is not None and button in self.mouse[name]):
                return name
        return None

    def handle_event(self, event, active=True):
        """Feed every pygame event. Returns 'quit', 'next_round' or None for control keys.
        Outside `active` play only mouse releases and focus changes are tracked, so no stale button state is kept."""
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.WINDOWFOCUSGAINED:
            self._ignore_mouse_until = self._clock() + FOCUS_GRACE
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.clear()
        elif event.type == pygame.MOUSEBUTTONUP:
            self._held_mouse.discard(event.button)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if active and self._clock() >= self._ignore_mouse_until:
                name = self._name_of(button=event.button)
                if name is not None:
                    self._held_mouse.add(event.button)
                    self._latched = name
        elif event.type == pygame.KEYDOWN and active:
            if event.key in self.bind["quit"]:
                return "quit"
            if event.key in self.bind["next_round"]:
                return "next_round"
            name = self._name_of(key=event.key)
            if name is not None:
                self._latched = name
        return None

    def poll(self, pressed=None):
        """Action for this tick. `pressed` is a key-state mapping (defaults to pygame's)."""
        pressed = pygame.key.get_pressed() if pressed is None else pressed
        name, self._latched = self._latched, None
        if name is None:
            for cand in _HELD_PRIORITY:
                if any(pressed[k] for k in self.bind[cand]) or any(b in self._held_mouse for b in self.mouse[cand]):
                    name = cand
                    break
        return int(_ACTIONS[name]) if name else int(Action.IDLE)

    def clear(self):
        self._latched = None
        self._held_mouse.clear()

"""Keyboard -> Action. Taps are latched until the next game tick; held keys are re-read every tick."""
import pygame

from adaptive_npc.env.actions import Action

_ACTION_KEYS = ("up", "down", "left", "right", "attack", "block", "dodge")
_ACTIONS = {"up": Action.UP, "down": Action.DOWN, "left": Action.LEFT, "right": Action.RIGHT,
            "attack": Action.ATTACK, "block": Action.BLOCK, "dodge": Action.DODGE}
# When several held keys compete: commit actions beat movement.
_HELD_PRIORITY = ("dodge", "attack", "block", "up", "down", "left", "right")


class KeyboardInput:
    def __init__(self, keys_cfg):
        self.bind = {name: [pygame.key.key_code(k) for k in keys_cfg[name]]
                     for name in list(_ACTION_KEYS) + ["next_round", "quit"]}
        self._latched = None

    def _name_of(self, key):
        for name in _ACTION_KEYS:
            if key in self.bind[name]:
                return name
        return None

    def handle_event(self, event):
        """Feed every pygame event. Returns 'quit', 'next_round' or None for control keys."""
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key in self.bind["quit"]:
                return "quit"
            if event.key in self.bind["next_round"]:
                return "next_round"
            name = self._name_of(event.key)
            if name is not None:
                self._latched = name
        return None

    def poll(self, pressed=None):
        """Action for this tick. `pressed` is a key-state mapping (defaults to pygame's)."""
        pressed = pygame.key.get_pressed() if pressed is None else pressed
        name, self._latched = self._latched, None
        if name is None:
            for cand in _HELD_PRIORITY:
                if any(pressed[k] for k in self.bind[cand]):
                    name = cand
                    break
        return int(_ACTIONS[name]) if name else int(Action.IDLE)

    def clear(self):
        self._latched = None

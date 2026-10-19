"""Game controller -> Action, with the same buffering as the keyboard: a button tap or a stick flick is latched until
the next game tick, held buttons (block) are re-read every tick. Uses pygame's SDL game-controller layer, so buttons
are named by physical position (a = south, b = east, y = north) and Xbox / PlayStation pads behave the same.
Presses only count while `active` (the PLAY state); releases and axis motion are always tracked."""
import pygame

from adaptive_npc.env.actions import Action
from adaptive_npc.render.input import _ACTIONS, _HELD_PRIORITY

BUTTON_POSITION = {"a": "south", "b": "east", "x": "west", "y": "north"}
AXIS_X, AXIS_Y = pygame.CONTROLLER_AXIS_LEFTX, pygame.CONTROLLER_AXIS_LEFTY
DPAD = {"up": "dpad_up", "down": "dpad_down", "left": "dpad_left", "right": "dpad_right"}


def button_code(name):
    try:
        return getattr(pygame, "CONTROLLER_BUTTON_" + name.upper())
    except AttributeError:
        raise ValueError(f"unknown controller button {name!r}") from None


def norm_axis(value):
    """SDL reports axes as ints in -32768..32767; floats are taken as already normalised."""
    v = value / 32768.0 if isinstance(value, int) else float(value)
    return max(-1.0, min(1.0, v))


def button_label(cfg, name):
    """On-screen name of a face button, from the configured label set (xbox by default)."""
    return cfg["label_sets"][cfg["labels"]][BUTTON_POSITION.get(name, name)]


class StickDirection:
    """Stick (x, y) -> 'up' / 'down' / 'left' / 'right' / None on the dominant axis. Starts moving past `deadzone`,
    stops below deadzone - hysteresis, and only switches axis when the other axis leads by `margin`, so a stick held
    near a diagonal does not flicker between two directions."""

    def __init__(self, deadzone=0.5, hysteresis=0.1, margin=0.15):
        self.deadzone, self.hysteresis, self.margin = float(deadzone), float(hysteresis), float(margin)
        self.dir = None

    def update(self, x, y):
        ax, ay = abs(x), abs(y)
        release = self.deadzone - self.hysteresis
        cur = self.dir
        if max(ax, ay) < (self.deadzone if cur is None else release):
            self.dir = None
            return None
        vertical = ay > ax
        if cur is not None:
            cur_vertical = cur in ("up", "down")
            cur_mag, other = (ay, ax) if cur_vertical else (ax, ay)
            if vertical != cur_vertical and cur_mag >= release and other < cur_mag + self.margin:
                vertical = cur_vertical
        if vertical:
            self.dir = "down" if y > 0 else "up"
        else:
            self.dir = "right" if x > 0 else "left"
        return self.dir

    def reset(self):
        self.dir = None


class ControllerInput:
    def __init__(self, cfg):
        self.bind = {name: [button_code(b) for b in cfg[name]] for name in ("attack", "block", "dodge")}
        for d, button in DPAD.items():
            self.bind[d] = [button_code(button)]
        self.stick = StickDirection(cfg["deadzone"], cfg["hysteresis"], cfg["axis_margin"])
        self._axes = {AXIS_X: 0.0, AXIS_Y: 0.0}
        self._latched = None
        self._held = set()

    def _name_of(self, button):
        for name in _HELD_PRIORITY:
            if button in self.bind[name]:
                return name
        return None

    def handle_event(self, event, active=True):
        t = event.type
        if t == pygame.CONTROLLERAXISMOTION and event.axis in self._axes:
            self._axes[event.axis] = norm_axis(event.value)
            prev = self.stick.dir
            now = self.stick.update(self._axes[AXIS_X], self._axes[AXIS_Y])
            if active and now is not None and now != prev:
                self._latched = now
        elif t == pygame.CONTROLLERBUTTONUP:
            self._held.discard(event.button)
        elif t == pygame.CONTROLLERBUTTONDOWN and active:
            name = self._name_of(event.button)
            if name is not None:
                self._held.add(event.button)
                self._latched = name
        elif t == pygame.WINDOWFOCUSLOST:
            self.release_all()
        return None

    def poll(self):
        name, self._latched = self._latched, None
        if name is None:
            for cand in _HELD_PRIORITY:
                if any(b in self._held for b in self.bind[cand]):
                    name = cand
                    break
            else:
                name = self.stick.dir
        return int(_ACTIONS[name]) if name else int(Action.IDLE)

    def clear(self):
        """Drop taps and held buttons (match start, resume). A stick that is still pushed keeps moving."""
        self._latched = None
        self._held.clear()

    def release_all(self):
        """Forget everything including the stick (focus lost, controller unplugged)."""
        self.clear()
        self._axes = {AXIS_X: 0.0, AXIS_Y: 0.0}
        self.stick.reset()


class ControllerHub:
    """Opens the connected game controllers and follows hot-plug events. `handle_event` returns 'connected' or
    'disconnected' when a controller came or went, else None. Without controller support everything stays inert."""

    def __init__(self):
        self.pads = {}   # instance id -> (Controller, name)
        try:
            from pygame._sdl2 import controller as sdl
            sdl.init()
        except Exception:   # pygame build without the controller module
            self._sdl = None
            return
        self._sdl = sdl
        for index in range(sdl.get_count()):
            self._open(index)

    def _open(self, index):
        sdl = self._sdl
        if not sdl.is_controller(index):
            return False
        pad = sdl.Controller(index)
        iid = pad.as_joystick().get_instance_id()
        if iid in self.pads:
            return False
        self.pads[iid] = (pad, sdl.name_forindex(index) or "CONTROLLER")
        return True

    @property
    def connected(self):
        return bool(self.pads)

    @property
    def name(self):
        return next(iter(self.pads.values()))[1] if self.pads else ""

    def handle_event(self, event):
        if self._sdl is None:
            return None
        if event.type == pygame.CONTROLLERDEVICEADDED:
            return "connected" if self._open(event.device_index) else None
        if event.type == pygame.CONTROLLERDEVICEREMOVED:
            if self.pads.pop(event.instance_id, None) is not None:
                return "disconnected"
        return None

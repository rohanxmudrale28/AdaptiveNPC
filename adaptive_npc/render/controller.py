"""Game controller -> Action, with the same buffering as the keyboard: a button tap or a stick flick is latched until
the next game tick, held buttons (block) are re-read every tick. Uses pygame's SDL game-controller layer, so buttons
are named by physical position (a = south, b = east, y = north) and Xbox / PlayStation pads behave the same. Pads
without an SDL mapping fall back to pygame.joystick (ControllerHub.translate) and a one-time binding screen.
Presses only count while `active` (the PLAY state); releases and axis motion are always tracked."""
from pathlib import Path

import pygame
import yaml

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


JOY_EVENTS = (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP, pygame.JOYAXISMOTION, pygame.JOYHATMOTION)
MAP_KEYS = ("attack", "block", "dodge", "pause", "axis_x", "axis_y", "invert_x", "invert_y")
BIND_STEPS = ("attack", "block", "dodge", "axis_x", "axis_y", "pause")
BIND_BUTTONS = ("attack", "block", "dodge", "pause")
BIND_PROMPTS = {"attack": "PRESS THE BUTTON FOR ATTACK", "block": "PRESS THE BUTTON FOR BLOCK",
                "dodge": "PRESS THE BUTTON FOR DASH", "axis_x": "PUSH THE LEFT STICK RIGHT",
                "axis_y": "PUSH THE LEFT STICK DOWN", "pause": "PRESS THE START BUTTON (PAUSE)"}
BIND_NAMES = {"attack": "ATTACK", "block": "BLOCK", "dodge": "DASH", "axis_x": "STICK X", "axis_y": "STICK Y",
              "pause": "START"}
HAT_DIRS = ("up", "down", "left", "right")


def hat_pressed(value):
    """Hat position (x, y) -> set of D-pad directions held (pygame hats: y = +1 is up)."""
    x, y = value
    return {d for d, on in (("up", y > 0), ("down", y < 0), ("left", x < 0), ("right", x > 0)) if on}


class _Pad:
    def __init__(self, kind, name, handle, mapping=None):
        self.kind, self.name, self.handle, self.map = kind, name, handle, mapping   # kind: controller | joystick
        self.hats = {}                                                               # hat index -> directions held


class BindingFlow:
    """One-time binding of a generic pad: press the buttons for ATTACK, BLOCK, DASH, push the stick right and down,
    press START. Fed raw JOY* events of that pad; `skip()` keeps the current binding of a step. Pure logic, no IO."""

    def __init__(self, instance_id, name, current, threshold=0.6):
        self.iid, self.name, self.map, self.threshold = instance_id, name, dict(current), float(threshold)
        self.step, self.note = 0, ""
        self.captured = set()

    @property
    def done(self):
        return self.step >= len(BIND_STEPS)

    @property
    def current(self):
        return None if self.done else BIND_STEPS[self.step]

    @property
    def prompt(self):
        return "" if self.done else BIND_PROMPTS[self.current]

    def describe(self, step):
        """Current value of a step for display: 'BUTTON 3', 'AXIS 0' (+ ' INVERTED')."""
        if step in BIND_BUTTONS:
            return f"BUTTON {self.map[step]}"
        return f"AXIS {self.map[step]}" + (" INVERTED" if self.map["invert_" + step[-1]] else "")

    def _advance(self, note=""):
        self.step += 1
        self.note = note

    def skip(self):
        if not self.done:
            self.captured.discard(self.current)
            self._advance("SKIPPED")

    def feed(self, event):
        if self.done or getattr(event, "instance_id", None) != self.iid:
            return
        cur = self.current
        if cur in BIND_BUTTONS:
            if event.type != pygame.JOYBUTTONDOWN:
                return
            if any(self.map[r] == event.button for r in self.captured):
                self.note = f"BUTTON {event.button} IS ALREADY USED"
                return
            self.map[cur] = event.button
        else:
            if event.type != pygame.JOYAXISMOTION or abs(norm_axis(event.value)) < self.threshold:
                return
            if cur == "axis_y" and "axis_x" in self.captured and event.axis == self.map["axis_x"]:
                return   # still the same stick motion as the previous step
            self.map[cur] = event.axis
            self.map["invert_" + cur[-1]] = norm_axis(event.value) < 0
        self.captured.add(cur)
        self._advance()


class ControllerHub:
    """Opens the connected pads and follows hot-plug events. A pad with an SDL game-controller mapping is a
    'controller' (its events are already in the controller vocabulary). Any other pad that pygame.joystick sees is a
    generic 'joystick': `translate()` turns its raw button / axis / hat events into controller events using a binding
    map (defaults from render.yaml, per-pad overrides from the local file), so the rest of the input code is shared.
    `handle_event` returns 'connected' / 'disconnected' when a pad came or went, else None."""

    def __init__(self, cfg=None):
        self.pads = {}   # instance id -> _Pad
        self.cfg = cfg or {}
        self.joy = self.cfg.get("joystick")   # None: no generic-pad fallback
        self._sdl = None
        try:
            from pygame._sdl2 import controller as sdl
            sdl.init()
            self._sdl = sdl
        except Exception:   # pygame build without the controller module
            pass
        if self.joy is not None:
            pygame.joystick.init()
        if pygame.joystick.get_init():
            count = pygame.joystick.get_count()
        else:
            count = self._sdl.get_count() if self._sdl else 0
        for index in range(count):
            self._open(index)

    # -- opening / bindings
    def _add(self, iid, pad):
        if iid in self.pads:
            return False
        self.pads[iid] = pad
        return True

    def _open(self, index):
        sdl = self._sdl
        if sdl is not None and sdl.is_controller(index):   # name_forindex is only safe for mapped pads
            pad = sdl.Controller(index)
            return self._add(pad.as_joystick().get_instance_id(),
                             _Pad("controller", sdl.name_forindex(index) or "CONTROLLER", pad))
        if self.joy is None:
            return False
        joystick = pygame.joystick.Joystick(index)
        name = joystick.get_name() or "GENERIC PAD"
        return self._add(joystick.get_instance_id(), _Pad("joystick", name, joystick, self.map_for(name)))

    def _local(self):
        path = Path(self.joy["local_file"])
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        except (OSError, yaml.YAMLError):
            data = None
        pads = data.get("pads") if isinstance(data, dict) else None
        return pads if isinstance(pads, dict) else {}

    def map_for(self, name):
        """Binding map of a generic pad: render.yaml defaults overridden by what was saved for this pad name."""
        saved = self._local().get(name)
        saved = {k: v for k, v in saved.items() if k in MAP_KEYS} if isinstance(saved, dict) else {}
        return {**self.joy["default"], **saved}

    def save_map(self, name, mapping):
        """Write the binding of the pad called `name` to the local file and apply it to the connected pads."""
        pads = self._local()
        pads[name] = {k: mapping[k] for k in MAP_KEYS}
        path = Path(self.joy["local_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Written by the controller binding screen (per pad name). Not in version control.\n"
                        + yaml.safe_dump({"pads": pads}, sort_keys=False), encoding="utf-8")
        for pad in self.pads.values():
            if pad.kind == "joystick" and pad.name == name:
                pad.map = dict(pads[name])

    def start_binding(self):
        """A BindingFlow for the first connected generic pad, or None."""
        for iid, pad in self.pads.items():
            if pad.kind == "joystick":
                return BindingFlow(iid, pad.name, pad.map, self.joy.get("bind_threshold", 0.6))
        return None

    # -- state
    @property
    def connected(self):
        return bool(self.pads)

    @property
    def name(self):
        return next(iter(self.pads.values())).name if self.pads else ""

    @property
    def generic(self):
        """True when a generic (joystick-fallback) pad is connected, so the binding screen applies."""
        return any(p.kind == "joystick" for p in self.pads.values())

    def is_generic_event(self, event):
        """A raw JOY* event of a generic pad (the binding screen consumes these)."""
        pad = self.pads.get(getattr(event, "instance_id", None)) if event.type in JOY_EVENTS else None
        return pad is not None and pad.kind == "joystick"

    # -- events
    def handle_event(self, event):
        t = event.type
        if t in (pygame.CONTROLLERDEVICEADDED, pygame.JOYDEVICEADDED):
            return "connected" if self._open(event.device_index) else None
        if t in (pygame.CONTROLLERDEVICEREMOVED, pygame.JOYDEVICEREMOVED):
            return "disconnected" if self.pads.pop(event.instance_id, None) is not None else None
        return None

    def translate(self, event):
        """Events to process for `event`: itself, or for a generic pad the controller events its raw event means
        (none if the input is unbound). Raw JOY events of SDL-mapped controllers are dropped (their CONTROLLER events
        already arrive)."""
        if event.type not in JOY_EVENTS:
            return [event]
        pad = self.pads.get(getattr(event, "instance_id", None))
        if pad is None or pad.kind != "joystick":
            return []
        iid, m, c = event.instance_id, pad.map, self.cfg

        def button(kind, name):
            return pygame.event.Event(kind, button=button_code(name), instance_id=iid)

        if event.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
            kind = pygame.CONTROLLERBUTTONDOWN if event.type == pygame.JOYBUTTONDOWN else pygame.CONTROLLERBUTTONUP
            for role in BIND_BUTTONS:
                if m[role] == event.button:
                    return [button(kind, c[role][0])]
            return []
        if event.type == pygame.JOYAXISMOTION:
            for axis, out, inv in (("axis_x", AXIS_X, "invert_x"), ("axis_y", AXIS_Y, "invert_y")):
                if m[axis] == event.axis:
                    v = norm_axis(event.value)
                    return [pygame.event.Event(pygame.CONTROLLERAXISMOTION, axis=out, value=-v if m[inv] else v,
                                               instance_id=iid)]
            return []
        now = hat_pressed(event.value)       # JOYHATMOTION: the D-pad, as button presses and releases
        before = pad.hats.get(event.hat, set())
        pad.hats[event.hat] = now
        return ([button(pygame.CONTROLLERBUTTONUP, DPAD[d]) for d in HAT_DIRS if d in before - now]
                + [button(pygame.CONTROLLERBUTTONDOWN, DPAD[d]) for d in HAT_DIRS if d in now - before])

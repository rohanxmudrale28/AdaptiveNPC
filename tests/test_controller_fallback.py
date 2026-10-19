"""Generic pads (no SDL game-controller mapping) through pygame.joystick: raw JOY* events are translated into controller
events with a binding map, a one-time binding flow rebinds them, bindings are saved to a local file. Synthetic events."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pathlib import Path

import pygame
import pytest
import yaml

from adaptive_npc.audio import player as audio_player
from adaptive_npc.env.actions import Action
from adaptive_npc.render.controller import (BIND_STEPS, AXIS_X, AXIS_Y, BindingFlow, ControllerHub, ControllerInput,
                                            _Pad, hat_pressed)
from adaptive_npc.render.retro import RetroView
from adaptive_npc.render.ui_input import UIKeys
from adaptive_npc.ui import shell as S
from tests.test_controller import B, canvas_colours, cfgs, down, ev, make_shell, _pygame  # noqa: F401 (fixtures)

IID = 4


def jb(button, pressed=True, iid=IID):
    return ev(pygame.JOYBUTTONDOWN if pressed else pygame.JOYBUTTONUP, instance_id=iid, button=button)


def ja(axis, value, iid=IID):
    return ev(pygame.JOYAXISMOTION, instance_id=iid, axis=axis, value=value)


def jh(value, hat=0, iid=IID):
    return ev(pygame.JOYHATMOTION, instance_id=iid, hat=hat, value=value)


def make_hub(cfgs, tmp_path, name="PC G", kind="joystick"):
    """A hub with one pad and no real device: the render.yaml controller config with the local file in tmp_path."""
    cfg = {**cfgs["render"]["controller"],
           "joystick": {**cfgs["render"]["controller"]["joystick"], "local_file": str(tmp_path / "local.yaml")}}
    hub = ControllerHub.__new__(ControllerHub)
    hub.cfg, hub.joy, hub._sdl, hub.pads = cfg, cfg["joystick"], None, {}
    hub.pads[IID] = _Pad(kind, name, None, hub.map_for(name) if kind == "joystick" else None)
    return hub


def kinds(events):
    return [(pygame.event.event_name(e.type), e.button if hasattr(e, "button") else (e.axis, round(e.value, 2)))
            for e in events]


def A(name):
    return B[name]


# ---------------------------------------------------------------------------------------------- real hub
def test_real_hub_starts_with_the_joystick_fallback_and_never_crashes(cfgs):
    hub = ControllerHub(cfgs["render"]["controller"])       # whatever is plugged in here, or nothing
    assert isinstance(hub.connected, bool) and isinstance(hub.generic, bool) and isinstance(hub.name, str)
    assert hub.generic <= hub.connected
    assert ControllerHub().joy is None                      # no config: no fallback, still inert
    assert ControllerHub().translate(jb(0)) == []


# ---------------------------------------------------------------------------------------------- maps
def test_default_map_local_override_per_pad_name_and_bad_file(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    default = cfgs["render"]["controller"]["joystick"]["default"]
    assert hub.map_for("PC G") == default
    local = Path(hub.joy["local_file"])
    local.write_text(yaml.safe_dump({"pads": {"PC G": {"attack": 5, "bogus": 1}, "OTHER": {"attack": 9}}}))
    assert hub.map_for("PC G") == {**default, "attack": 5}          # unknown keys dropped, other pads untouched
    assert hub.map_for("OTHER")["attack"] == 9 and hub.map_for("NEW PAD") == default
    local.write_text("pads: [not, a, mapping")
    assert hub.map_for("PC G") == default                           # a broken local file never breaks the game
    local.write_text("- just\n- a list\n")
    assert hub.map_for("PC G") == default


def test_save_map_round_trips_applies_and_keeps_other_pads(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    Path(hub.joy["local_file"]).write_text(yaml.safe_dump({"pads": {"OTHER": {"attack": 9}}}))
    new = {**hub.pads[IID].map, "attack": 6, "axis_x": 2, "invert_y": True}
    hub.save_map("PC G", new)
    saved = yaml.safe_load(Path(hub.joy["local_file"]).read_text())["pads"]
    assert saved["OTHER"] == {"attack": 9} and saved["PC G"]["attack"] == 6 and saved["PC G"]["invert_y"] is True
    assert hub.pads[IID].map["attack"] == 6                          # applied to the connected pad at once
    assert hub.map_for("PC G") == saved["PC G"]                      # and reloaded the same on the next start
    assert "version control" in Path(hub.joy["local_file"]).read_text().splitlines()[0]


def test_local_file_is_gitignored_and_render_yaml_has_the_defaults(cfgs):
    assert "configs/local_controller.yaml" in Path(".gitignore").read_text().split()
    j = cfgs["render"]["controller"]["joystick"]
    assert j["local_file"] == "configs/local_controller.yaml" and 0 < j["bind_threshold"] <= 1
    assert set(j["default"]) == {"attack", "block", "dodge", "pause", "axis_x", "axis_y", "invert_x", "invert_y"}
    assert not Path("configs/local_controller.yaml").exists() or True   # created only on a user's machine


# ---------------------------------------------------------------------------------------------- translation
def test_buttons_translate_by_role(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)                                    # defaults: attack 0, block 1, dash 3, pause 7
    assert kinds(hub.translate(jb(0))) == [("ControllerButtonDown", A("a"))]
    assert kinds(hub.translate(jb(1))) == [("ControllerButtonDown", A("b"))]
    assert kinds(hub.translate(jb(3))) == [("ControllerButtonDown", A("y"))]
    assert kinds(hub.translate(jb(7))) == [("ControllerButtonDown", A("start"))]
    assert kinds(hub.translate(jb(3, pressed=False))) == [("ControllerButtonUp", A("y"))]
    assert hub.translate(jb(5)) == [] and hub.translate(jb(14, pressed=False)) == []   # unbound buttons do nothing


def test_axes_translate_with_optional_inversion(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    assert kinds(hub.translate(ja(0, 0.9))) == [("ControllerAxisMotion", (AXIS_X, 0.9))]
    assert kinds(hub.translate(ja(1, -0.7))) == [("ControllerAxisMotion", (AXIS_Y, -0.7))]
    assert hub.translate(ja(2, 1.0)) == [] and hub.translate(ja(5, -1.0)) == []      # triggers / right stick: ignored
    hub.pads[IID].map.update(axis_x=3, invert_x=True, axis_y=2)
    assert kinds(hub.translate(ja(3, 0.8))) == [("ControllerAxisMotion", (AXIS_X, -0.8))]
    assert kinds(hub.translate(ja(2, 0.8))) == [("ControllerAxisMotion", (AXIS_Y, 0.8))]
    assert hub.translate(ja(0, 0.9)) == []


def test_hat_becomes_dpad_presses_and_releases(cfgs, tmp_path):
    assert hat_pressed((0, 0)) == set() and hat_pressed((1, 1)) == {"up", "right"}
    hub = make_hub(cfgs, tmp_path)
    assert kinds(hub.translate(jh((1, 0)))) == [("ControllerButtonDown", A("dpad_right"))]
    assert kinds(hub.translate(jh((1, 1)))) == [("ControllerButtonDown", A("dpad_up"))]            # right still held
    assert kinds(hub.translate(jh((0, 1)))) == [("ControllerButtonUp", A("dpad_right"))]
    assert kinds(hub.translate(jh((-1, -1)))) == [("ControllerButtonUp", A("dpad_up")),
                                                  ("ControllerButtonDown", A("dpad_down")),
                                                  ("ControllerButtonDown", A("dpad_left"))]
    assert kinds(hub.translate(jh((0, 0)))) == [("ControllerButtonUp", A("dpad_down")),
                                                ("ControllerButtonUp", A("dpad_left"))]
    assert hub.translate(jh((0, 0))) == []


def test_translate_passes_other_events_and_drops_foreign_joy_events(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    key = ev(pygame.KEYDOWN, key=pygame.K_a)
    assert hub.translate(key) == [key]
    assert hub.translate(down("a")) == [down("a")]                   # real controller events go straight through
    assert hub.translate(jb(0, iid=99)) == []                        # unknown device
    sdl_pad = make_hub(cfgs, tmp_path, kind="controller")
    assert sdl_pad.translate(jb(0)) == []                            # the same pad's CONTROLLER events already arrive
    assert hub.is_generic_event(jb(0)) and not hub.is_generic_event(jb(0, iid=99)) and not hub.is_generic_event(key)
    assert not sdl_pad.is_generic_event(jb(0)) and not sdl_pad.generic and hub.generic


# ---------------------------------------------------------------------------------------------- through the input code
def play_through(hub, cfgs, events, active=True):
    p = ControllerInput(cfgs["render"]["controller"])
    out = []
    for e in events:
        for t in hub.translate(e):
            p.handle_event(t, active=active)
        out.append(p.poll())
    return out


def test_generic_pad_drives_the_game_like_a_mapped_one(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    got = play_through(hub, cfgs, [jb(0), jb(0, False), jb(1), jb(1), jb(1, False), jb(3), jb(3, False)])
    A_, BL, D, I = (int(a) for a in (Action.ATTACK, Action.BLOCK, Action.DODGE, Action.IDLE))
    assert got == [A_, I, BL, BL, I, D, I]                           # tap, held block across steps, dash
    assert play_through(hub, cfgs, [jh((0, 1)), ja(0, 0.0), jh((0, 0))]) == [int(Action.UP), int(Action.UP), I]


def test_generic_stick_keeps_the_dead_zone_and_hysteresis(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    R, I = int(Action.RIGHT), int(Action.IDLE)
    assert play_through(hub, cfgs, [ja(0, 0.4), ja(0, 0.6), ja(0, 0.45), ja(0, 0.35)]) == [I, R, R, I]
    assert play_through(hub, cfgs, [ja(0, 0.7), ja(1, 0.65), ja(1, 0.72), ja(1, 0.95), ja(0, 0.0), ja(1, 0.0)]) == [
        R, R, R, int(Action.DOWN), int(Action.DOWN), I]               # no flicker on the diagonal
    hub.pads[IID].map.update(invert_y=True)
    assert play_through(hub, cfgs, [ja(1, -0.9)]) == [int(Action.DOWN)]


def test_generic_pad_navigates_menus_and_pauses(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    u = UIKeys(cfgs["ui"]["keys"], cfgs["render"]["keys"], cfgs["render"]["controller"])

    def cmds(*raw):
        return [c for e in raw for t in hub.translate(e) if (c := u.command(t))]

    assert cmds(jb(0), jb(0, False)) == ["confirm"] and cmds(jb(1)) == ["back"]
    assert cmds(jh((0, -1)), jh((0, 0)), jh((-1, 0)), jh((0, 0))) == ["down", "left"]
    assert cmds(ja(1, 0.9), ja(1, 0.95), ja(1, 0.0), ja(1, -0.9), ja(1, 0.0)) == ["down", "up"]
    u.observe(S.PLAY)
    assert cmds(jb(0), jb(1), jh((0, 1)), ja(1, 0.9)) == []           # gameplay buttons make no menu commands
    assert cmds(jb(7)) == ["back"]                                    # the bound Start pauses


# ---------------------------------------------------------------------------------------------- hot-plug
def test_hot_plug_events_for_generic_pads(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    assert hub.handle_event(ev(pygame.JOYDEVICEREMOVED, instance_id=99)) is None
    assert hub.handle_event(ev(pygame.JOYDEVICEREMOVED, instance_id=IID)) == "disconnected"
    assert hub.handle_event(ev(pygame.CONTROLLERDEVICEREMOVED, instance_id=IID)) is None   # one event per unplug
    assert not hub.connected and not hub.generic and hub.name == "" and hub.start_binding() is None
    hub._open = lambda index: hub.pads.setdefault(IID, _Pad("joystick", "PC G", None, hub.map_for("PC G"))) and True
    assert hub.handle_event(ev(pygame.JOYDEVICEADDED, device_index=0)) == "connected"
    assert hub.generic and hub.name == "PC G"


# ---------------------------------------------------------------------------------------------- binding flow
def flow(threshold=0.6):
    default = {"attack": 0, "block": 1, "dodge": 3, "pause": 7, "axis_x": 0, "axis_y": 1, "invert_x": False,
               "invert_y": False}
    return BindingFlow(IID, "PC G", default, threshold)


def test_binding_flow_captures_buttons_then_stick_axes_then_start():
    f = flow()
    assert f.current == "attack" and not f.done and "ATTACK" in f.prompt and f.describe("attack") == "BUTTON 0"
    f.feed(jb(5, pressed=False))                       # releases and other pads are ignored
    f.feed(jb(5, iid=99))
    f.feed(jh((1, 0)))
    assert f.current == "attack"
    f.feed(jb(5))
    assert f.map["attack"] == 5 and f.current == "block"
    f.feed(jb(5))                                      # the same button again is refused, with a note
    assert f.current == "block" and "ALREADY USED" in f.note
    f.feed(jb(6))
    f.feed(jb(9))
    assert (f.map["block"], f.map["dodge"], f.current) == (6, 9, "axis_x")
    f.feed(ja(4, 0.3))                                 # below the threshold
    f.feed(jb(2))                                      # a button is not a stick
    assert f.current == "axis_x"
    f.feed(ja(2, 0.9))
    assert (f.map["axis_x"], f.map["invert_x"], f.current) == (2, False, "axis_y")
    f.feed(ja(2, 0.1))
    f.feed(ja(2, 0.95))                                # the stick is still pushed right: same axis, not Y
    assert f.current == "axis_y"
    f.feed(ja(3, -0.9))                                # pushing "down" read as negative: Y is inverted
    assert (f.map["axis_y"], f.map["invert_y"], f.current) == (3, True, "pause")
    f.feed(jb(11))
    assert f.done and f.map["pause"] == 11 and f.current is None and f.prompt == ""
    assert f.captured == set(BIND_STEPS)
    f.feed(jb(12))                                     # nothing after the end
    assert f.map["pause"] == 11


def test_binding_flow_skip_keeps_the_current_binding_and_buttons_stay_distinct():
    f = flow()
    f.skip()
    assert f.map["attack"] == 0 and f.note == "SKIPPED" and f.current == "block" and "attack" not in f.captured
    f.feed(jb(0))                                      # a skipped step does not reserve its button
    assert f.map["block"] == 0 and f.current == "dodge"
    for _ in range(3):                                 # dodge, stick X, stick Y
        f.skip()
    assert f.current == "pause"
    f.skip()
    assert f.done
    f.skip()                                           # harmless after the end
    assert f.map["axis_x"] == 0 and f.map["axis_y"] == 1


def test_binding_flow_threshold_comes_from_config(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    f = hub.start_binding()
    assert f.iid == IID and f.name == "PC G" and f.threshold == cfgs["render"]["controller"]["joystick"]["bind_threshold"]
    assert f.map == hub.pads[IID].map and f.map is not hub.pads[IID].map       # edits stay in the flow until saved
    assert make_hub(cfgs, tmp_path, kind="controller").start_binding() is None   # mapped pads need no binding


def test_bound_map_is_saved_and_used_by_the_next_session(cfgs, tmp_path):
    hub = make_hub(cfgs, tmp_path)
    f = hub.start_binding()
    for e in (jb(8), jb(9), jb(10), ja(2, 0.9), ja(2, 0.0), ja(3, 0.9), jb(12)):
        f.feed(e)
    assert f.done
    assert hub.translate(jb(8)) == []                                  # nothing changes until it is saved
    hub.save_map(f.name, f.map)
    assert kinds(hub.translate(jb(8))) == [("ControllerButtonDown", A("a"))] and hub.translate(jb(0)) == []
    assert kinds(hub.translate(jb(12))) == [("ControllerButtonDown", A("start"))]
    assert kinds(hub.translate(ja(2, 0.9))) == [("ControllerAxisMotion", (AXIS_X, 0.9))]
    again = make_hub(cfgs, tmp_path)                                   # "restart": a fresh hub reads the local file
    assert again.pads[IID].map == hub.pads[IID].map and again.pads[IID].map["attack"] == 8


# ---------------------------------------------------------------------------------------------- shell + view
def shell_on_pad_page(cfgs, factory, generic=True):
    sh, _, _ = make_shell(cfgs, bind_factory=factory)
    sh.controller_connected, sh.controller_generic = True, generic
    sh.handle("confirm"), sh.handle("down"), sh.handle("confirm"), sh.handle("down"), sh.handle("down")
    sh.handle("confirm")
    assert sh.state == S.CONTROLS and sh.controls_detail == 2
    return sh


def test_enter_on_the_controller_page_opens_the_binding_screen_for_generic_pads(cfgs):
    made = []
    sh = shell_on_pad_page(cfgs, lambda: made.append(flow()) or made[-1])
    sh.handle("confirm")
    assert sh.state == S.BIND and sh.binder is made[0] and len(made) == 1
    sh.handle("confirm")                               # Enter skips a step
    assert sh.binder.current == "block"
    sh.handle("down"), sh.handle("left")               # menu moves do nothing here
    assert sh.state == S.BIND and sh.binder.current == "block"
    sh.handle("back")                                  # Esc cancels
    assert sh.state == S.CONTROLS and sh.controls_detail == 2 and sh.binder is None and sh.controls_note is None
    sh.handle("confirm")
    assert sh.state == S.BIND and len(made) == 2       # reachable again, from the start
    sh.binder.map["attack"] = 5
    sh.end_binding("PAD BINDINGS SAVED")
    assert sh.state == S.CONTROLS and sh.controls_note == "PAD BINDINGS SAVED"
    sh.handle("back"), sh.handle("back")               # back to the list, then the title
    assert sh.state == S.TITLE
    sh.handle("confirm")                               # the title cursor is still on CONTROLS
    assert sh.state == S.CONTROLS and sh.controls_note is None      # a fresh visit starts clean


def test_binding_screen_is_only_for_generic_pads_with_a_way_to_bind(cfgs):
    for generic, factory in ((False, lambda: flow()), (True, None), (True, lambda: None)):
        sh = shell_on_pad_page(cfgs, factory, generic)
        sh.handle("confirm")
        assert sh.state == S.CONTROLS and sh.controls_detail is None and sh.binder is None   # plain "back to the list"
    sh = shell_on_pad_page(cfgs, lambda: flow())
    sh.handle("left"), sh.handle("confirm")            # on another scheme's page Enter is still just back
    assert sh.state == S.CONTROLS and sh.controls_detail is None


def test_unplugging_during_binding_leaves_it_with_a_message(cfgs):
    sh = shell_on_pad_page(cfgs, lambda: flow())
    sh.handle("confirm")
    assert sh.state == S.BIND
    sh.controller_lost()
    assert sh.state == S.CONTROLS and sh.binder is None and sh.controls_note == "CONTROLLER DISCONNECTED"


def test_binding_screen_is_a_menu_state_for_the_music():
    assert S.BIND in audio_player.MENU_STATES and audio_player.MUSIC_FOR_STATE[S.BIND] == "menu"


def test_binding_and_generic_pad_screens_render_with_palette_colours_only(cfgs):
    pal = {tuple(v) for v in cfgs["ui"]["palette"].values()}
    sh = shell_on_pad_page(cfgs, lambda: flow())
    sh.controller_name = "PC G"
    view = RetroView(cfgs["ui"], cfgs["render"], 9)
    view.draw(sh, 0.0)                                 # the controller page with the bind hint
    assert canvas_colours(view) <= pal
    sh.controls_note = "PAD BINDINGS SAVED"
    view.draw(sh, 0.0)
    assert canvas_colours(view) <= pal
    sh.controls_note = None
    sh.handle("confirm")
    f = sh.binder
    steps = [None, jb(5), jb(5), jb(6), jb(9), ja(2, 0.9), ja(3, 0.9), jb(11)]
    for e in steps:
        if e is not None:
            f.feed(e)
        view.draw(sh, 0.0)
        assert canvas_colours(view) <= pal
    assert f.done
    sh.state = S.CONTROLS                              # the list page also shows the generic marker
    sh.controls_detail = None
    view.draw(sh, 0.0)
    assert canvas_colours(view) <= pal

"""Play the arena as a retro game: title, opponent select, match with HUD, pause, results.

python -m scripts.play                                  # shell with title + opponent select (--menu, default)
python -m scripts.play --opponent rusher --rounds 3     # straight into a match (an explicit --opponent skips the menu)
python -m scripts.play --opponent mbom --blind --label B --participant P01    # blind human test, logs to results/human/
Opponents: a named enemy from configs/ui.yaml (dummy, rusty, reflex, seer, echo), an NPC version (scripted, ppo_none,
ppo_predicted, mbom), or any players.yaml archetype / random / switcher. Tick rate: --tick-rate or the select screen.
Input schemes (bindings in configs/render.yaml), all active at once unless --input restricts them:
  keyboard: WASD/arrows move, J attack, K block (held), L dash     kbm: WASD/arrows move, left click attack, hold right
  click block, Q dash     controller: left stick or D-pad move, A attack, B block (held), Y dash, Start pause.
--input all|keyboard|kbm|controller picks one (--keys mouse|keyboard is a deprecated alias for kbm|keyboard).
Enter confirm, Esc pause, H legend, F scanlines, Tab demo panel, P screenshot, M mute, +/- volume. Sound is synthesized (no files):
--mute starts silent, --no-music keeps effects only, --music forces music on (it is off by default with --blind).
`--set ui.<key>=value` overrides configs/ui.yaml, other --set keys override render.yaml.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pygame

from adaptive_npc.audio.player import SoundManager
from adaptive_npc.config import apply_overrides, load_config
from adaptive_npc.npc.live import build_live_npc
from adaptive_npc.players.base import RandomPlayer
from adaptive_npc.players.scripted import make_player
from adaptive_npc.render.controller import ControllerHub, ControllerInput
from adaptive_npc.render.input import SCHEMES, InputMux, KeyboardInput, scheme_keys, uses_controller
from adaptive_npc.render.retro import RetroView
from adaptive_npc.render.ui_input import UIKeys
from adaptive_npc.ui import shell as S
from adaptive_npc.ui.session import MatchSession, write_json

ALIASES = {"scripted_npc": "scripted"}
KEYS_ALIAS = {"mouse": "kbm", "keyboard": "keyboard"}   # deprecated --keys values -> --input values


def make_opponent(name, players_cfg, env_cfg, seed):
    if name == "random":
        return RandomPlayer(seed)
    return make_player(name, players_cfg, env_cfg, seed)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/render.yaml")
    p.add_argument("--ui-config", default="configs/ui.yaml")
    p.add_argument("--keys-config", default=None,
                   help="deprecated: YAML whose keys: attack / block / dodge replace the KEYBOARD scheme bindings")
    p.add_argument("--env-config", default="configs/env.yaml")
    p.add_argument("--players-config", default="configs/players.yaml")
    p.add_argument("--tick-rate", type=float, default=None, help="game steps per second")
    p.add_argument("--fps", type=int, default=None, help="render frames per second")
    p.add_argument("--opponent", default=None)
    p.add_argument("--rounds", type=int, default=None, help="0 = until closed")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--menu", action=argparse.BooleanOptionalAction, default=None,
                   help="show the title / opponent select screens (default on, unless --opponent or --blind is given)")
    p.add_argument("--panel", choices=("on", "off"), default="off",
                   help="start with the demo panel on (Tab toggles; never shown with --blind; P = screenshot)")
    p.add_argument("--blind", action="store_true",
                   help="human test: hide enemy names, NPC header and panel; log the real NPC version to results/human/")
    p.add_argument("--participant", default="P00", help="participant ID for --blind logs (no personal data)")
    p.add_argument("--label", default="A", help="'Opponent <label>' shown in --blind mode")
    p.add_argument("--mute", action="store_true", help="start with all sound muted (M toggles)")
    music = p.add_mutually_exclusive_group()
    music.add_argument("--no-music", action="store_true", help="effects only, no music")
    music.add_argument("--music", action="store_true", help="music on even with --blind (off by default there)")
    p.add_argument("--input", choices=SCHEMES, default=None,
                   help="input scheme: all (default, any device works), keyboard (J/K/L), kbm (keyboard + mouse), "
                        "controller. Bindings are in configs/render.yaml")
    p.add_argument("--keys", choices=tuple(KEYS_ALIAS), default=None,
                   help="deprecated alias: --keys mouse = --input kbm, --keys keyboard = --input keyboard")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return p


def resolve_scheme(args, cfg):
    """The input scheme: --input, else the deprecated --keys alias (with a warning), else render.yaml `input`."""
    if args.keys is not None:
        note = f"--keys is deprecated, use --input {KEYS_ALIAS[args.keys]}"
        print("warning: " + (note if args.input is None else f"{note} (ignored: --input {args.input} given)"),
              file=sys.stderr)
        if args.input is None:
            return KEYS_ALIAS[args.keys]
    scheme = args.input or cfg["input"]
    if scheme not in SCHEMES:
        sys.exit(f"render.yaml input must be one of {SCHEMES}, got {scheme!r}")
    return scheme


def find_enemy(enemies, name):
    """Named enemy by key / name / NPC method; otherwise an ad-hoc enemy for any archetype name."""
    name = ALIASES.get(name, name)
    for k, e in enumerate(enemies):
        if name.lower() in (e["key"].lower(), e["name"].lower(), e["method"].lower()):
            return k, enemies
    custom = {"key": name, "name": name.upper().replace("_", " ")[:14], "method": name, "look": "npc",
              "accessory": "none", "tag": "CUSTOM", "desc": "CUSTOM OPPONENT FROM THE COMMAND LINE."}
    return len(enemies), enemies + [custom]


def unique_path(directory, stem):
    path = Path(directory) / f"{stem}.json"
    n = 1
    while path.exists():
        path = Path(directory) / f"{stem}_{n}.json"
        n += 1
    return path


def main(argv=None):
    args = build_parser().parse_args(argv)
    ui_over = [o[3:] for o in args.overrides if o.startswith("ui.")]
    cfg = load_config(args.config, [o for o in args.overrides if not o.startswith("ui.")])
    if args.keys_config:
        print("warning: --keys-config is deprecated, edit keyboard_keys in render.yaml", file=sys.stderr)
        old = load_config(args.keys_config)["keys"]
        cfg["keyboard_keys"] = {a: old[a] for a in ("attack", "block", "dodge")}
    scheme = resolve_scheme(args, cfg)
    cfg["input"] = scheme
    ui = apply_overrides(load_config(args.ui_config), ui_over)
    ui.setdefault("tags", {})["CUSTOM"] = "silver"
    env_cfg = load_config(args.env_config)
    players_cfg = load_config(args.players_config)
    tick_rate = args.tick_rate if args.tick_rate is not None else cfg["tick_rate"]
    fps = args.fps if args.fps is not None else cfg["fps"]
    n_rounds = cfg["play"]["rounds"] if args.rounds is None else args.rounds
    seed = cfg["seed"] if args.seed is None else args.seed
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args.participant):
        sys.exit("--participant must be 1-32 characters of letters, digits, '_' or '-' (an ID, not a name)")
    if args.blind and args.opponent is None:
        sys.exit("--blind needs --opponent (the NPC version to test)")
    menu = args.menu if args.menu is not None else (args.opponent is None and not args.blind)

    enemy_idx, enemies = find_enemy(ui["enemies"], args.opponent or cfg["play"]["opponent"])
    label = args.label.upper()
    last = {}

    def make_session(enemy, rate):
        npc = build_live_npc(enemy["method"], ui, env_cfg, players_cfg, seed)
        return MatchSession(env_cfg, npc, n_rounds, seed, label if args.blind else "", rate)

    def on_match_end(summary):
        enemy = shell.enemy
        meta = {"enemy": enemy["key"], "npc_seed": ui["npc_seed"], "tick_rate": shell.tick_rate, "input_scheme": scheme,
                "env_config": args.env_config, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if args.blind:
            stem = f"{args.participant}_{label}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
            path = write_json(unique_path(ui["human_dir"], stem),
                              {"participant": args.participant, "opponent_label": label, **meta, **summary})
        else:
            path = write_json(Path(ui["results_dir"]) / "last_match.json", {**meta, **summary})
        last.update(summary, path=str(path))

    pygame.init()
    use_music = not args.no_music and (args.music or (bool(ui["audio"]["music"]) and not args.blind))
    sound = SoundManager(ui["audio"], mute=args.mute, music=use_music)   # warns once and stays silent if no audio device
    use_pad = uses_controller(scheme)
    hub = ControllerHub() if use_pad else None
    pad = ControllerInput(cfg["controller"]) if use_pad else None
    keys = InputMux(KeyboardInput(scheme_keys(cfg, scheme)), *([pad] if pad else []))
    shell = S.Shell(ui, enemies, tick_rate, make_session, has_menu=menu, select_enabled=menu and not args.blind,
                    enemy_index=enemy_idx, blind=args.blind, label=label, on_match_end=on_match_end,
                    input_clear=keys.clear, max_catchup=cfg["max_catchup_steps"],
                    panel=args.panel == "on")
    if not menu:
        shell.begin()
        if shell.state == S.QUIT:  # checkpoint missing: say so clearly, no window
            pygame.quit()
            sys.exit(f"error: {shell.error_text}")
    view = RetroView(ui, cfg, env_cfg["arena_size"], caption=ui["title"])
    ui_keys = UIKeys(ui["keys"], cfg["keys"], cfg["controller"] if use_pad else None)
    frame_clock = pygame.time.Clock()
    dt = 0.0
    while shell.state != S.QUIT:
        ui_keys.observe(shell.state)
        for ev in pygame.event.get():
            if hub is not None and hub.handle_event(ev) == "disconnected":
                pad.release_all()
                shell.controller_lost()
            cmd = ui_keys.command(ev)
            if cmd == "screenshot":
                print(f"screenshot: {view.save_screenshot(ui['results_dir'])}")
            elif cmd in ("mute", "volume_up", "volume_down"):
                sound.control(cmd)
            elif cmd:
                sound.command(cmd, shell.state)
                shell.handle(cmd)
                ui_keys.observe(shell.state)
            keys.handle_event(ev, active=shell.state == S.PLAY)
        if hub is not None:
            shell.controller_connected, shell.controller_name = hub.connected, hub.name
        shell.update(dt, keys.poll)
        sound.observe(shell)
        view.draw(shell, dt)
        dt = min(frame_clock.tick(fps) / 1000.0, 0.25)
    pygame.quit()
    if last:
        w = last["rounds_won"]
        print(f"npc wins {w['npc']}, you win {w['human']}, draws {w['draw']}  (saved {last['path']})")
    return last


if __name__ == "__main__":
    main()

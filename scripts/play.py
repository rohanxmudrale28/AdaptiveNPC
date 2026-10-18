"""Play the arena as a retro game: title, opponent select, match with HUD, pause, results.

python -m scripts.play                                  # shell with title + opponent select (--menu, default)
python -m scripts.play --opponent rusher --rounds 3     # straight into a match (an explicit --opponent skips the menu)
python -m scripts.play --opponent mbom --blind --label B --participant P01    # blind human test, logs to results/human/
Opponents: a named enemy from configs/ui.yaml (dummy, rusty, reflex, seer, echo), an NPC version (scripted, ppo_none,
ppo_predicted, mbom), or any players.yaml archetype / random / switcher. Tick rate: --tick-rate or the select screen.
Controls are in configs/render.yaml (WASD/arrows move, J attack, K block, L dodge, Enter confirm, Esc pause,
H legend, F scanlines, Tab demo panel, P screenshot). `--set ui.<key>=value` overrides configs/ui.yaml, other --set keys override render.yaml.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pygame

from adaptive_npc.config import apply_overrides, load_config
from adaptive_npc.npc.live import build_live_npc
from adaptive_npc.players.base import RandomPlayer
from adaptive_npc.players.scripted import make_player
from adaptive_npc.render.input import KeyboardInput
from adaptive_npc.render.retro import RetroView
from adaptive_npc.render.ui_input import UIKeys
from adaptive_npc.ui import shell as S
from adaptive_npc.ui.session import MatchSession, write_json

ALIASES = {"scripted_npc": "scripted"}


def make_opponent(name, players_cfg, env_cfg, seed):
    if name == "random":
        return RandomPlayer(seed)
    return make_player(name, players_cfg, env_cfg, seed)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/render.yaml")
    p.add_argument("--ui-config", default="configs/ui.yaml")
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
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return p


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
        meta = {"enemy": enemy["key"], "npc_seed": ui["npc_seed"], "tick_rate": shell.tick_rate,
                "env_config": args.env_config, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if args.blind:
            stem = f"{args.participant}_{label}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
            path = write_json(unique_path(ui["human_dir"], stem),
                              {"participant": args.participant, "opponent_label": label, **meta, **summary})
        else:
            path = write_json(Path(ui["results_dir"]) / "last_match.json", {**meta, **summary})
        last.update(summary, path=str(path))

    pygame.init()
    keys = KeyboardInput(cfg["keys"])
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
    ui_keys = UIKeys(ui["keys"], cfg["keys"])
    frame_clock = pygame.time.Clock()
    dt = 0.0
    while shell.state != S.QUIT:
        for ev in pygame.event.get():
            cmd = ui_keys.command(ev)
            if cmd == "screenshot":
                print(f"screenshot: {view.save_screenshot(ui['results_dir'])}")
            elif cmd:
                shell.handle(cmd)
            if shell.state == S.PLAY:
                keys.handle_event(ev)
        shell.update(dt, keys.poll)
        view.draw(shell, dt)
        dt = min(frame_clock.tick(fps) / 1000.0, 0.25)
    pygame.quit()
    if last:
        w = last["rounds_won"]
        print(f"npc wins {w['npc']}, you win {w['human']}, draws {w['draw']}  (saved {last['path']})")


if __name__ == "__main__":
    main()

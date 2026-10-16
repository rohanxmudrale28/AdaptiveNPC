"""Human (player 1, blue) vs a scripted opponent / random (player 0, orange).

python -m scripts.play --tick-rate 8 [--opponent rusher] [--rounds 5] [--fps 60]
Controls are in configs/render.yaml (WASD/arrows move, J attack, K block, L dodge, Enter next round, Esc quit).
"""
import argparse

import numpy as np
import pygame

from adaptive_npc.config import load_config
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.observations import encode
from adaptive_npc.players.base import RandomPlayer, Transition
from adaptive_npc.players.scripted import make_player
from adaptive_npc.render.clock import TickClock
from adaptive_npc.render.input import KeyboardInput
from adaptive_npc.render.view import ArenaView


def make_opponent(name, players_cfg, env_cfg, seed):
    if name == "random":
        return RandomPlayer(seed)
    return make_player(name, players_cfg, env_cfg, seed)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/render.yaml")
    p.add_argument("--env-config", default="configs/env.yaml")
    p.add_argument("--players-config", default="configs/players.yaml")
    p.add_argument("--tick-rate", type=float, default=None, help="game steps per second")
    p.add_argument("--fps", type=int, default=None, help="render frames per second")
    p.add_argument("--opponent", default=None)
    p.add_argument("--rounds", type=int, default=None, help="0 = until closed")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    env_cfg = load_config(args.env_config)
    players_cfg = load_config(args.players_config)
    tick_rate = args.tick_rate if args.tick_rate is not None else cfg["tick_rate"]
    fps = args.fps if args.fps is not None else cfg["fps"]
    opp_name = args.opponent or cfg["play"]["opponent"]
    n_rounds = cfg["play"]["rounds"] if args.rounds is None else args.rounds
    seed = cfg["seed"] if args.seed is None else args.seed

    pygame.init()
    view = ArenaView(cfg, env_cfg["arena_size"], title=f"Adaptive NPC - you vs {opp_name} @ {tick_rate:g} steps/s")
    keys = KeyboardInput(cfg["keys"])
    clock = TickClock(tick_rate, cfg["max_catchup_steps"])
    frame_clock = pygame.time.Clock()

    game = ArenaGame(env_cfg, rng=np.random.default_rng(seed))
    opp = make_opponent(opp_name, players_cfg, env_cfg, seed)
    opp.seed(seed)
    names = (opp_name, "YOU")
    score = [0, 0, 0]  # opponent wins, human wins, draws
    rnd, running = 0, True

    while running and (n_rounds == 0 or rnd < n_rounds):
        game.reset()
        opp.reset()
        keys.clear()
        clock.reset()
        hist0, hist1 = [], []
        done = False
        while running and not done:
            for ev in pygame.event.get():
                if keys.handle_event(ev) == "quit":
                    running = False
            for _ in range(clock.advance(frame_clock.get_time() / 1000.0)):
                o0, o1 = encode(game, 0), encode(game, 1)
                a0 = int(opp.act(o0, hist0))
                a1 = keys.poll()
                r, done = game.step(a0, a1)
                hist0.append((a0, a1))
                hist1.append((a1, a0))
                if opp.learns:
                    opp.observe(Transition(o0, a0, a1, r, encode(game, 0), done))
                if done:
                    break
            view.draw(game, score, rnd, names)
            frame_clock.tick(fps)
        if not running:
            break
        score[{0: 0, 1: 1, -1: 2}[game.winner]] += 1
        verdict = {0: f"{opp_name} wins", 1: "YOU WIN", -1: "DRAW"}[game.winner]
        last = n_rounds != 0 and rnd + 1 >= n_rounds
        prompt = "Esc to quit" if last else "Enter: next round   Esc: quit"
        waiting = True
        while waiting and running:
            for ev in pygame.event.get():
                act = keys.handle_event(ev)
                if act == "quit":
                    running = False
                elif act == "next_round" and not last:
                    waiting = False
            view.draw(game, score, rnd, names, banner=[verdict, prompt])
            frame_clock.tick(fps)
        rnd += 1
    pygame.quit()
    print(f"opponent wins {score[0]}, you win {score[1]}, draws {score[2]}")


if __name__ == "__main__":
    main()

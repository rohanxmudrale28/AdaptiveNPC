"""Multi-round match runner. Players keep their state across rounds (reset() is per round)."""
import numpy as np

from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.observations import encode
from adaptive_npc.players.base import Transition


class MatchHook:
    """Override any method to observe a match (logging, switching archetypes, rendering)."""

    def on_round_start(self, round_idx, game, npc, player):
        pass

    def on_step(self, round_idx, game, a_npc, a_player):
        pass

    def on_round_end(self, round_idx, game, result):
        pass


def run_match(cfg, npc, player, n_rounds=30, seed=0, hook=None):
    """Play n_rounds rounds of npc (player 0) vs player (player 1).

    Returns one dict per round: round, winner (0 npc / 1 player / -1 draw), hp_npc,
    hp_player, steps, ret_npc (NPC return incl. shaping).
    """
    hook = hook or MatchHook()
    game = ArenaGame(cfg, rng=np.random.default_rng(seed))
    npc.seed(seed)
    player.seed(seed + 1)
    results = []
    for rnd in range(n_rounds):
        game.reset()
        npc.reset()
        player.reset()
        hist0, hist1, ret = [], [], 0.0
        hook.on_round_start(rnd, game, npc, player)
        done = False
        while not done:
            o0, o1 = encode(game, 0), encode(game, 1)
            a0 = int(npc.act(o0, hist0))
            a1 = int(player.act(o1, hist1))
            r, done = game.step(a0, a1)
            hist0.append((a0, a1))
            hist1.append((a1, a0))
            ret += r
            if npc.learns:
                npc.observe(Transition(o0, a0, a1, r, encode(game, 0), done))
            if player.learns:
                player.observe(Transition(o1, a1, a0, -r, encode(game, 1), done))
            hook.on_step(rnd, game, a0, a1)
        result = {"round": rnd, "winner": game.winner, "hp_npc": game.hp[0],
                  "hp_player": game.hp[1], "steps": game.t, "ret_npc": ret}
        results.append(result)
        hook.on_round_end(rnd, game, result)
    return results


def npc_win_rate(results):
    """NPC win rate with draws counted as 1/2."""
    score = sum(1.0 if r["winner"] == 0 else 0.5 if r["winner"] == -1 else 0.0 for r in results)
    return score / len(results)

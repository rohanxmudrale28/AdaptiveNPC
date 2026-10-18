"""MatchSession: one human-vs-NPC match on an ArenaGame, with the statistics the results screen shows.

The session is the only thing that steps the game. Player 0 = NPC (orange), player 1 = human (blue). Scores are
[NPC wins, human wins, draws].
"""
import json
from pathlib import Path

import numpy as np

from adaptive_npc.env.core import ArenaGame

WINNER_NAME = {0: "npc", 1: "human", -1: "draw"}


class MatchSession:
    def __init__(self, env_cfg, npc, rounds, seed, label="", tick_rate=None):
        self.game = ArenaGame(env_cfg, rng=np.random.default_rng(seed))
        self.npc, self.n_rounds, self.seed = npc, int(rounds), seed
        self.label, self.tick_rate = label, tick_rate
        self.score = [0, 0, 0]
        self.round = 0              # index of the current / next round
        self.rounds = []            # one dict per finished round
        self.dmg_dealt = 0          # by the human
        self.dmg_taken = 0          # by the human
        self.pred_hits = self.pred_steps = 0
        self._r = None
        self.last_pred = self.last_actual = None  # the NPC's guess for the last step and what the human did
        self.start_round()

    # ------------------------------------------------------------------ rounds
    def start_round(self):
        self.game.reset()
        self.npc.reset_round()
        self.last_pred = self.last_actual = None
        self._r = {"steps": 0, "dealt": 0, "taken": 0, "hits": 0, "pred_steps": 0}

    @property
    def finished(self):
        return self.n_rounds != 0 and self.round >= self.n_rounds

    @property
    def round_done(self):
        return self.game.done

    def next_round(self):
        """Call after a finished round has been recorded (tick() does that). Starts the next one."""
        self.round += 1
        if not self.finished:
            self.start_round()

    def tick(self, a_human):
        """One game step. Returns True when the round ended (and was recorded)."""
        g = self.game
        a_npc = int(self.npc.act(g))
        self.last_pred = None if self.npc.pred is None else np.array(self.npc.pred)
        self.last_actual = int(a_human)
        if self.npc.pred is not None:
            self._r["pred_steps"] += 1
            self._r["hits"] += int(np.argmax(self.npc.pred) == int(a_human))
        r, done = g.step(a_npc, int(a_human))
        self.npc.observe(a_npc, int(a_human), g, r, done)
        self._r["steps"] += 1
        self._r["dealt"] += g.dealt[1]
        self._r["taken"] += g.dealt[0]
        if done:
            self._record_round()
        return done

    def _record_round(self):
        g, r = self.game, self._r
        self.npc.end_round()
        self.score[{0: 0, 1: 1, -1: 2}[g.winner]] += 1
        self.dmg_dealt += r["dealt"]
        self.dmg_taken += r["taken"]
        self.pred_hits += r["hits"]
        self.pred_steps += r["pred_steps"]
        self._r = None  # folded into the match totals
        self.rounds.append({"round": self.round, "winner": WINNER_NAME[g.winner], "hp_npc": g.hp[0],
                            "hp_human": g.hp[1], "steps": r["steps"], "damage_dealt": r["dealt"],
                            "damage_taken": r["taken"],
                            "pred_accuracy": r["hits"] / r["pred_steps"] if r["pred_steps"] else None})

    # ------------------------------------------------------------------ results
    @property
    def pred_accuracy(self):
        return self.pred_hits / self.pred_steps if self.pred_steps else None

    @property
    def running_accuracy(self):
        """Prediction accuracy over the match so far, including the round in progress."""
        n = self.pred_steps + (self._r["pred_steps"] if self._r else 0)
        return (self.pred_hits + (self._r["hits"] if self._r else 0)) / n if n else None

    def verdict(self):
        """'human' / 'npc' / 'draw' by rounds won so far."""
        if self.score[1] > self.score[0]:
            return "human"
        return "npc" if self.score[0] > self.score[1] else "draw"

    def summary(self, completed):
        return {"npc_method": self.npc.method, "label": self.label, "seed": self.seed, "tick_rate": self.tick_rate,
                "completed": bool(completed), "rounds_planned": self.n_rounds, "rounds_played": len(self.rounds),
                "rounds_won": {"npc": self.score[0], "human": self.score[1], "draw": self.score[2]},
                "damage_dealt": self.dmg_dealt, "damage_taken": self.dmg_taken,
                "prediction_accuracy": self.pred_accuracy, "prediction_steps": self.pred_steps,
                "rounds": self.rounds}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path

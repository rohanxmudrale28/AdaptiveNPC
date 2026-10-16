"""Diagnose how counter_puncher wins: parries, damage sources, and results against two hand-written
test players (flanker, waiter) that probe whether the rules allow a counter to block-and-punish.

python -m scripts.diagnose_counter_puncher --config configs/balance.yaml

Read-only: no game or player code is modified; the test players are defined here.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from adaptive_npc.config import config_from_args, parse_args
from adaptive_npc.config import load_config
from adaptive_npc.env.actions import DIRS
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.observations import encode
from adaptive_npc.eval.balance import make_any
from adaptive_npc.players.scripted import ATTACK, IDLE, Scripted

CAUSES = ["target stunned (post-parry)", "target on attack cooldown", "flank (target not facing)",
          "target out of stamina", "other (frontal, unblocked)"]


class Waiter(Scripted):
    """Stays at distance 2 out of reach until the opponent's stamina is empty (so it cannot block)."""

    def act(self, obs, hist):
        self.opp_sta = obs[15]
        return super().act(obs, hist)

    def policy(self, v, hist):
        empty = self.opp_sta <= 0
        if v.dist >= 3 or (v.dist == 2 and empty):
            return v.toward()
        if v.dist == 2:
            return IDLE
        if not v.facing_opp:
            return v.toward()
        if empty and v.can_attack:
            return ATTACK
        return v.away() if not empty else IDLE


def classify(g, attacker, target, pre):
    """Cause of a hit by `attacker` on `target`, using the pre-step snapshot."""
    if pre["stun"][target] > 0:
        return CAUSES[0]
    if pre["cd"][target] > 0:
        return CAUSES[1]
    tf = DIRS[pre["face"][target]]
    if (pre["x"][target] + tf[0], pre["y"][target] + tf[1]) != (pre["x"][attacker], pre["y"][attacker]):
        return CAUSES[2]
    if pre["sta"][target] < 1:
        return CAUSES[3]
    return CAUSES[4]


def run_rounds(env_cfg, subject, other, subject_slot, n_rounds, seed):
    """Play n rounds with `subject` in `subject_slot`; return stats from the subject's view."""
    g = ArenaGame(env_cfg, rng=np.random.default_rng(seed))
    players = [None, None]
    players[subject_slot], players[1 - subject_slot] = subject, other
    subject.seed(seed + 1)
    other.seed(seed + 2)
    s, o = subject_slot, 1 - subject_slot
    st = Counter()
    dmg_by, dmg_taken = Counter(), Counter()
    for _ in range(n_rounds):
        g.reset()
        for p in players:
            p.reset()
        hist = [[], []]
        done = False
        while not done:
            obs = [encode(g, 0), encode(g, 1)]
            a = [int(players[i].act(obs[i], hist[i])) for i in (0, 1)]
            pre = {k: list(getattr(g, k)) for k in ("x", "y", "face", "sta", "cd", "stun")}
            _, done = g.step(*a)
            hist[0].append((a[0], a[1]))
            hist[1].append((a[1], a[0]))
            for att, tgt, book in ((s, o, dmg_by), (o, s, dmg_taken)):
                if g.dealt[att]:
                    book[classify(g, att, tgt, pre)] += g.dealt[att]
            st["parried_subject_attacks"] += g.parried[s]
            st["parried_other_attacks"] += g.parried[o]
        st["win" if g.winner == s else "loss" if g.winner == o else "draw"] += 1
        st["rounds"] += 1
    return st, dmg_by, dmg_taken


def main():
    cfg = config_from_args(parse_args(description=__doc__))
    env_cfg = load_config(cfg["env_config"])
    pcfg = load_config(cfg["players_config"])
    n = int(cfg["rounds_per_pair"])
    others = list(pcfg["train"]) + list(pcfg["heldout"]) + ["scripted_npc", "random", "waiter"]
    lines = [f"counter_puncher diagnosis: {n} rounds/side/seed, seeds {cfg['seeds']}, both sides pooled",
             "win%/draw%/loss% are counter_puncher's; parries are per round; damage is per round.", ""]
    tot_by, tot_taken = Counter(), Counter()
    for name in others:
        st, by, taken = Counter(), Counter(), Counter()
        for seed in cfg["seeds"]:
            for slot in (0, 1):
                sd = int(seed) * 7919 + slot
                cp = make_any("counter_puncher", pcfg, env_cfg, sd + 1000)
                if name == "waiter":
                    other = Waiter(pcfg["kiter"], env_cfg["arena_size"], sd + 2000)
                    other.noise = 0.0
                else:
                    other = make_any(name, pcfg, env_cfg, sd + 2000)
                a, b, c = run_rounds(env_cfg, cp, other, slot, n, sd)
                st += a; by += b; taken += c
        r = st["rounds"]
        lines.append(f"vs {name:15s} win {st['win']/r:5.1%} draw {st['draw']/r:5.1%} loss {st['loss']/r:5.1%}"
                     f" | CP parries {st['parried_other_attacks']/r:5.2f}/rd, CP attacks parried "
                     f"{st['parried_subject_attacks']/r:5.2f}/rd | dmg dealt {sum(by.values())/r:4.2f} "
                     f"taken {sum(taken.values())/r:4.2f}")
        if name not in ("flanker", "waiter"):
            tot_by += by
        if name in ("flanker", "waiter"):
            for lab, book in (("CP deals", by), ("CP takes", taken)):
                lines.append(f"      {lab}: " + ", ".join(f"{k.split(' (')[0]} {v/r:.2f}" for k, v in book.items()))
    lines += ["", "Damage the counter_puncher DEALS to the non-probe opponents, by cause (total hits):"]
    tot = sum(tot_by.values()) or 1
    for k in CAUSES:
        lines.append(f"  {k:32s} {tot_by[k]:6d} ({tot_by[k]/tot:5.1%})")
    report = "\n".join(lines)
    print(report)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis_counter_puncher.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""ArenaGame: pure-python rules for the 1v1 arena. No pygame, no gym, no rendering.

Both players act simultaneously. Per step, resolution order is
dodge -> move -> block -> attack (see PLAN.md decisions 2-4).
State is stored as 2-element lists indexed by player (0 = NPC, 1 = opponent).

Timers: a cooldown/stun of N set at step t makes the action unavailable for steps
t+1..t+N; the stored value counts down to 0 and the action is available when it is 0.
"""
import numpy as np

from adaptive_npc.env.actions import Action, DIRS

IDLE, ATTACK, BLOCK, DODGE = int(Action.IDLE), int(Action.ATTACK), int(Action.BLOCK), int(Action.DODGE)


class ArenaGame:
    def __init__(self, cfg, rng=None):
        self.size = int(cfg["arena_size"])
        self.max_steps = int(cfg["max_steps"])
        self.random_start = bool(cfg["random_start"])
        self.hp_max = int(cfg["hp"])
        self.sta_max = int(cfg["stamina"]["max"])
        self.regen_every = int(cfg["stamina"]["regen_every"])
        self.atk_damage = int(cfg["attack"]["damage"])
        self.atk_cooldown = int(cfg["attack"]["cooldown"])
        self.block_cost = int(cfg["block"]["stamina_cost"])
        self.parry_stun = int(cfg["block"]["parry_stun"])
        self.dodge_cost = int(cfg["dodge"]["stamina_cost"])
        self.dodge_dist = int(cfg["dodge"]["distance"])
        rw = cfg["reward"]
        self.r_win, self.r_loss, self.r_draw = rw["win"], rw["loss"], rw["draw"]
        self.r_dealt, self.r_taken = rw["damage_dealt"], rw["damage_taken"]
        self.rng = rng if rng is not None else np.random.default_rng(cfg.get("seed", 0))
        self.reset()

    def reset(self):
        n = self.size
        if self.random_start:
            half = n // 2
            x0 = int(self.rng.integers(0, half))
            x1 = n - 1 - int(self.rng.integers(0, half))
            y0, y1 = (int(v) for v in self.rng.integers(0, n, size=2))
        else:
            x0, x1, y0, y1 = 1, n - 2, n // 2, n // 2
        self.x = [x0, x1]
        self.y = [y0, y1]
        self.face = [3, 2]  # p0 faces right, p1 faces left
        self.hp = [self.hp_max, self.hp_max]
        self.sta = [self.sta_max, self.sta_max]
        self.cd = [0, 0]
        self.stun = [0, 0]
        self.t = 0
        self.done = False
        self.winner = None  # 0, 1, or -1 (draw) once done
        self.dealt = [0, 0]  # damage dealt by each player in the last step
        self.parried = [False, False]  # player i's attack was parried in the last step
        self.blocking = [False, False]
        self.invuln = [False, False]

    # ------------------------------------------------------------------ helpers
    def _free(self, x, y, ox, oy):
        return 0 <= x < self.size and 0 <= y < self.size and (x, y) != (ox, oy)

    def _dash_dest(self, i):
        j = 1 - i
        dx, dy = DIRS[self.face[i]]
        x, y = self.x[i], self.y[i]
        for _ in range(self.dodge_dist):
            if not self._free(x + dx, y + dy, self.x[j], self.y[j]):
                break
            x, y = x + dx, y + dy
        return x, y

    def front(self, i):
        dx, dy = DIRS[self.face[i]]
        return self.x[i] + dx, self.y[i] + dy

    # --------------------------------------------------------------------- step
    def step(self, a0, a1):
        """Advance one step. Returns (reward for player 0, done)."""
        acts = [int(a0), int(a1)]
        can_attack = [False, False]
        for i in (0, 1):
            if self.stun[i] > 0:
                acts[i] = IDLE
                self.stun[i] -= 1
            can_attack[i] = self.cd[i] == 0
            if self.cd[i] > 0:
                self.cd[i] -= 1

        # Actions the player cannot afford / is not ready for fall back to idle.
        for i in (0, 1):
            a = acts[i]
            if a == DODGE:
                if self.sta[i] >= self.dodge_cost:
                    self.sta[i] -= self.dodge_cost
                else:
                    acts[i] = IDLE
            elif a == BLOCK:
                if self.sta[i] >= self.block_cost:
                    self.sta[i] -= self.block_cost
                else:
                    acts[i] = IDLE
            elif a == ATTACK and not can_attack[i]:
                acts[i] = IDLE

        self.invuln = [acts[0] == DODGE, acts[1] == DODGE]
        self.blocking = [acts[0] == BLOCK, acts[1] == BLOCK]

        # 1. dodge dashes (simultaneous; identical destinations cancel)
        dests = {i: self._dash_dest(i) for i in (0, 1) if acts[i] == DODGE}
        if len(dests) == 2 and dests[0] == dests[1]:
            dests = {}
        for i, (x, y) in dests.items():
            self.x[i], self.y[i] = x, y

        # 2. moves (simultaneous; identical destinations cancel; facing always updates)
        mv = {}
        for i in (0, 1):
            a = acts[i]
            if Action.UP <= a <= Action.RIGHT:
                self.face[i] = a - 1
                dx, dy = DIRS[a - 1]
                j = 1 - i
                nx, ny = self.x[i] + dx, self.y[i] + dy
                if self._free(nx, ny, self.x[j], self.y[j]):
                    mv[i] = (nx, ny)
        if len(mv) == 2 and mv[0] == mv[1]:
            mv = {}
        for i, (x, y) in mv.items():
            self.x[i], self.y[i] = x, y

        # 3/4. attacks (block flags and i-frames were fixed above); all resolve simultaneously
        self.dealt = [0, 0]
        self.parried = [False, False]
        hits = []
        for i in (0, 1):
            if acts[i] != ATTACK:
                continue
            j = 1 - i
            self.cd[i] = self.atk_cooldown
            if self.front(i) != (self.x[j], self.y[j]) or self.invuln[j]:
                continue
            if self.blocking[j] and self.front(j) == (self.x[i], self.y[i]):
                self.parried[i] = True
            else:
                hits.append((i, j))
        for i, j in hits:
            self.hp[j] -= self.atk_damage
            self.dealt[i] += self.atk_damage
        for i in (0, 1):
            if self.parried[i]:
                self.stun[i] = self.parry_stun

        self.t += 1
        if self.t % self.regen_every == 0:
            for i in (0, 1):
                self.sta[i] = min(self.sta_max, self.sta[i] + 1)

        r = self.r_dealt * self.dealt[0] + self.r_taken * self.dealt[1]
        if self.hp[0] <= 0 or self.hp[1] <= 0 or self.t >= self.max_steps:
            self.done = True
            if self.hp[0] > self.hp[1]:
                self.winner, r = 0, r + self.r_win
            elif self.hp[1] > self.hp[0]:
                self.winner, r = 1, r + self.r_loss
            else:
                self.winner, r = -1, r + self.r_draw
        return r, self.done

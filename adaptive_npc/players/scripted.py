"""Scripted archetypes. All read only their own-view observation (observations.encode), so each
can play either side. Parameters come from configs/players.yaml.

Observation indices used: 0,1 own pos | 2,3 rel pos to opponent | 4..7 own facing | 8..11 opp facing
| 14 own stamina | 16,17 own/opp attack cooldown | 18,19 own/opp stun.
"""
import numpy as np

from adaptive_npc.env.actions import Action, DIRS, N_ACTIONS
from adaptive_npc.players.base import Player

IDLE, ATTACK, BLOCK, DODGE = int(Action.IDLE), int(Action.ATTACK), int(Action.BLOCK), int(Action.DODGE)
MOVE_OF_FACING = [f + 1 for f in range(len(DIRS))]  # facing f <-> move action f+1


class View:
    """Decoded integer view of an observation vector."""

    def __init__(self, obs, span):
        self.x = int(round(obs[0] * span))
        self.y = int(round(obs[1] * span))
        self.dx = int(round(obs[2] * span))
        self.dy = int(round(obs[3] * span))
        self.face = int(np.argmax(obs[4:8]))
        self.opp_face = int(np.argmax(obs[8:12]))
        self.has_stamina = obs[14] > 0
        self.can_attack = obs[16] == 0
        self.opp_cd = obs[17] > 0
        self.opp_stunned = obs[19] > 0
        self.span = span
        self.dist = abs(self.dx) + abs(self.dy)

    def dir_toward(self):
        """Facing index of the move that closes the larger axis gap."""
        if abs(self.dx) >= abs(self.dy):
            return 3 if self.dx > 0 else 2
        return 1 if self.dy > 0 else 0

    @property
    def adjacent(self):
        return self.dist == 1

    @property
    def facing_opp(self):
        """Opponent stands exactly on the tile in front of us."""
        return self.adjacent and DIRS[self.face] == (self.dx, self.dy)

    @property
    def opp_facing_us(self):
        return self.adjacent and DIRS[self.opp_face] == (-self.dx, -self.dy)

    def toward(self):
        return MOVE_OF_FACING[self.dir_toward()]

    def away(self):
        """Move that increases distance, avoiding walls; falls back to a sidestep."""
        d = self.dir_toward()
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}[d]
        side = (2, 3) if d in (0, 1) else (0, 1)
        for f in (opposite,) + side:
            nx, ny = self.x + DIRS[f][0], self.y + DIRS[f][1]
            if 0 <= nx <= self.span and 0 <= ny <= self.span:
                return MOVE_OF_FACING[f]
        return IDLE


class Scripted(Player):
    def __init__(self, params, arena_size, seed=None):
        self.params = params
        self.span = arena_size - 1
        self.noise = float(params.get("noise", 0.0))
        self.rng = np.random.default_rng(seed)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, hist):
        if self.noise > 0 and self.rng.random() < self.noise:
            return int(self.rng.integers(N_ACTIONS))
        return int(self.policy(View(obs, self.span), hist))

    def policy(self, v, hist):
        raise NotImplementedError


class Rusher(Scripted):
    """Charges the opponent and attacks whenever it can."""

    def policy(self, v, hist):
        if v.facing_opp:
            return ATTACK
        return v.toward()


class Camper(Scripted):
    """Holds its ground, faces the opponent, blocks when they come close, occasionally pokes."""

    def policy(self, v, hist):
        if v.dist > self.params["block_radius"]:
            return IDLE
        if v.adjacent and not v.facing_opp:
            return v.toward()  # turn to face
        if v.facing_opp and v.can_attack and self.rng.random() < self.params["poke_prob"]:
            return ATTACK
        return BLOCK if v.has_stamina else IDLE


class Repeater(Scripted):
    """Cycles through a fixed token pattern, ignoring the opponent except to resolve toward/away."""

    TOKENS = {"idle": IDLE, "up": int(Action.UP), "down": int(Action.DOWN), "left": int(Action.LEFT),
              "right": int(Action.RIGHT), "attack": ATTACK, "block": BLOCK, "dodge": DODGE}

    def policy(self, v, hist):
        tok = self.params["pattern"][len(hist) % len(self.params["pattern"])]
        if tok == "toward":
            return v.toward()
        if tok == "away":
            return v.away()
        return self.TOKENS[tok]


class Mixer(Scripted):
    """Samples each action i.i.d. from fixed weights."""

    def __init__(self, params, arena_size, seed=None):
        super().__init__(params, arena_size, seed)
        w = np.array([params["weights"][a.name.lower()] for a in Action], dtype=np.float64)
        self.p = w / w.sum()

    def policy(self, v, hist):
        return int(self.rng.choice(N_ACTIONS, p=self.p))


class Kiter(Scripted):
    """Hit and retreat: approach and strike, then back off while the attack is on cooldown."""

    def policy(self, v, hist):
        if not v.can_attack:
            return v.away()
        if v.facing_opp:
            return ATTACK
        return v.toward()


class CounterPuncher(Scripted):
    """Stalks to striking range, then blocks; punishes after the opponent attacks (or is stunned),
    reacting `reaction_delay` steps late."""

    def __init__(self, params, arena_size, seed=None):
        super().__init__(params, arena_size, seed)
        self.delay = int(params.get("reaction_delay", 0))
        self._trig = []

    def reset(self):
        self._trig = []

    def act(self, obs, hist):
        self._trig.append(bool(obs[17] > 0 or obs[19] > 0))  # opp on cooldown or stunned
        return super().act(obs, hist)

    def policy(self, v, hist):
        if v.dist > 2:
            return v.toward()
        if not v.adjacent:
            return BLOCK if v.has_stamina else IDLE
        if not v.facing_opp:
            return v.toward()
        punish = len(self._trig) > self.delay and self._trig[-1 - self.delay]
        if punish and v.can_attack:
            return ATTACK
        return BLOCK if v.has_stamina else IDLE


class Flanker(Scripted):
    """Walks to a tile two steps to the side of the opponent, steps in facing it, attacks from the
    side where a frontal block does not protect."""

    def policy(self, v, hist):
        if v.adjacent:
            if v.facing_opp and v.can_attack and not v.opp_facing_us:
                return ATTACK
            if v.opp_facing_us:
                return BLOCK if v.has_stamina else IDLE
            return v.toward() if not v.facing_opp else IDLE
        fx, fy = DIRS[v.opp_face]
        best = None
        for sgn in (1, -1):
            px, py = -fy * sgn, fx * sgn          # perpendicular to the opponent's facing
            rel = (v.dx + 2 * px, v.dy + 2 * py)  # stage tile relative to us
            if 0 <= v.x + rel[0] <= v.span and 0 <= v.y + rel[1] <= v.span:
                cand = (abs(rel[0]) + abs(rel[1]), rel, (px, py))
                best = cand if best is None or cand < best else best
        if best is None:
            return v.toward()
        _, rel, (px, py) = best
        if rel == (0, 0):                          # on the stage tile: step in facing the opponent
            return MOVE_OF_FACING[DIRS.index((-px, -py))]
        if abs(rel[0]) >= abs(rel[1]):
            return int(Action.RIGHT) if rel[0] > 0 else int(Action.LEFT)
        return int(Action.DOWN) if rel[1] > 0 else int(Action.UP)


class ScriptedNPC(Scripted):
    """Hand-coded baseline NPC: close in, attack when ready, block while the attack recharges."""

    def policy(self, v, hist):
        if v.facing_opp:
            if v.can_attack:
                return ATTACK
            return BLOCK if (v.opp_facing_us and v.has_stamina and not v.opp_cd) else IDLE
        return v.toward()


class Switcher(Player):
    """Cycles through sub-archetypes: every `switch_every_rounds` rounds, and optionally also
    at `switch_at_step` inside each round. reset() is called once per round."""

    def __init__(self, params, players, names):
        self.every = int(params["switch_every_rounds"])
        self.at_step = params.get("switch_at_step")
        self.players, self.names = players, names
        self.round = -1
        self._idx = 0

    def seed(self, seed):
        for i, p in enumerate(self.players):
            p.seed(seed + 1 + i)

    def reset(self):
        self.round += 1
        self._idx = (self.round // self.every) % len(self.players)
        for p in self.players:
            p.reset()

    def _index(self, t):
        shift = 1 if self.at_step is not None and t >= self.at_step else 0
        return (self._idx + shift) % len(self.players)

    @property
    def current_name(self):
        return self.names[self._idx]

    def act(self, obs, hist):
        return self.players[self._index(len(hist))].act(obs, hist)


ARCHETYPES = {"rusher": Rusher, "camper": Camper, "repeater": Repeater, "mixer": Mixer,
              "flanker": Flanker, "kiter": Kiter, "counter_puncher": CounterPuncher, "scripted_npc": ScriptedNPC}


def make_player(name, players_cfg, env_cfg, seed=None):
    """Build a scripted player by name from configs/players.yaml (+ arena size from env.yaml)."""
    size = env_cfg["arena_size"]
    seed = players_cfg.get("seed", 0) if seed is None else seed
    if name == "switcher":
        params = players_cfg["switcher"]
        names = list(params["cycle"])
        subs = [make_player(n, players_cfg, env_cfg, seed + 1 + i) for i, n in enumerate(names)]
        return Switcher(params, subs, names)
    return ARCHETYPES[name](players_cfg[name], size, seed)

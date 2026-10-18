"""16x16 pixel-grid sprites defined as character maps (no image files).

Two designs, a helmeted knight (human, blue) and a boxy robot (NPC, orange), recoloured per `look`:
  roles  . clear | k ink | a main | b shade | c highlight | e eyes/lights | m silver | n steel | w white | y gold
  (a, b, c, e come from the look; the rest are fixed palette entries)
Facings are the DIRS indices 0 up, 1 down, 2 left, 3 right (right = mirrored left). Frames (per facing):
  idle0 idle1 walk0 walk1 attack block ko
Dodge trails, the blinking of i-frames, stun stars and hit flashes are effects drawn by the view.
"""
import math

import pygame

SIZE = 16
BLANK = "." * SIZE
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
FIXED = {"k": "ink", "m": "silver", "n": "steel", "w": "white", "y": "amber"}

# ----------------------------------------------------------------------------------------------- knight
KNIGHT_TOP = {
    DOWN: [BLANK,
           "....kkkkkkkk....",
           "...kaaccccaak...",
           "...kaaaaaaaak...",
           "...kakwkkwkak...",
           "...kaaaaaaaak...",
           "....kbaaaabk....",
           "..kkaaccccaakk..",
           ".kakaaaccaaakak.",
           ".kakaayyyyaakak.",
           "..kkbbaaaabbkk..",
           "..kkbbaaaabbkk.."],
    UP: [BLANK,
         "....kkkkkkkk....",
         "...kaaaaaaaak...",
         "...kaabbbbaak...",
         "...kaabbbbaak...",
         "...kaaaaaaaak...",
         "....kbaaaabk....",
         "..kkaaaaaaaakk..",
         ".kakaabbbbaakak.",
         ".kakaabbbbaakak.",
         "..kkbbbbbbbbkk..",
         "..kkbbbbbbbbkk.."],
    LEFT: [BLANK,
           ".....kkkkkkk....",
           "....kaaccaaak...",
           "....kaaaaaaak...",
           "....kwkkaaaak...",
           "....kaaaaaaak...",
           ".....kbaaaabk...",
           "....kkaaccaakk..",
           "....kaaaccaaak..",
           "....kaayyyyaak..",
           "....kkbbaabbkk..",
           "....kkbbaabbkk.."],
}
KNIGHT_LEGS = {
    DOWN: {"stand": ["...kbbk..kbbk...", "...kbbk..kbbk...", "..kkkkk..kkkkk..", BLANK],
           "walk0": ["...kbbk..kbbk...", "...kkkk..kbbk...", ".........kkkkk..", BLANK],
           "walk1": ["...kbbk..kbbk...", "...kbbk..kkkk...", "..kkkkk.........", BLANK]},
    LEFT: {"stand": [".....kbbkbbk....", ".....kbbkbbk....", "....kkkkkkkkk...", BLANK],
           "walk0": [".....kbbkbbk....", "....kbbk.kbbk...", "...kkkk..kkkk...", BLANK],
           "walk1": [".....kbbkbbk....", ".....kbbk.kbbk..", ".....kkkk.kkkk..", BLANK]},
}
KNIGHT_LEGS[UP] = KNIGHT_LEGS[DOWN]

# weapon overlays: (x, y, role) per facing; the knight's sword and the robot's claw arm
KNIGHT_WEAPON = {
    DOWN: [(14, 9, "a"), (13, 10, "y"), (14, 10, "y"), (15, 10, "y")]
          + [(14, y, "m") for y in range(11, 15)] + [(15, y, "n") for y in range(11, 15)] + [(14, 15, "w"), (15, 15, "n")],
    UP: [(14, 8, "a"), (14, 9, "y"), (13, 7, "y"), (14, 7, "y"), (15, 7, "y")]
        + [(14, y, "m") for y in range(1, 7)] + [(15, y, "n") for y in range(1, 7)] + [(14, 0, "w")],
    LEFT: [(4, 9, "a"), (5, 9, "y"), (3, 8, "y"), (3, 9, "y"), (3, 10, "y")]
          + [(x, 9, "m") for x in range(0, 3)] + [(x, 10, "n") for x in range(0, 3)] + [(0, 9, "w")],
}

# ----------------------------------------------------------------------------------------------- robot
ROBOT_TOP = {
    DOWN: [".......ee.......",
           ".......kk.......",
           ".......kk.......",
           "...kkkkkkkkkk...",
           "...kaaaaaaaak...",
           "...kaeekkeeak...",
           "...kaaaaaaaak...",
           "...kbbbbbbbbk...",
           "..kkkkkkkkkkkk..",
           ".kakkaaaaaakkak.",
           ".kakkaccccakkak.",
           ".kbkkbbbbbbkkbk."],
    UP: [".......ee.......",
         ".......kk.......",
         ".......kk.......",
         "...kkkkkkkkkk...",
         "...kaaaaaaaak...",
         "...kaaaaaaaak...",
         "...kaabbbbaak...",
         "...kbbbbbbbbk...",
         "..kkkkkkkkkkkk..",
         ".kakkaaaaaakkak.",
         ".kakkabbbbakkak.",
         ".kbkkbbbbbbkkbk."],
    LEFT: [".......ee.......",
           ".......kk.......",
           ".......kk.......",
           "....kkkkkkkkk...",
           "....kaaaaaaak...",
           "....keekaaaak...",
           "....kaaaaaaak...",
           "....kbbbbbbbk...",
           "...kkkkkkkkkkk..",
           "...kakkaaaaak...",
           "...kakkaccccak..",
           "...kbkkbbbbbbk.."],
}
ROBOT_LEGS = KNIGHT_LEGS
ROBOT_WEAPON = {
    DOWN: [(14, y, "n") for y in range(10, 14)] + [(15, y, "m") for y in range(10, 14)] + [(14, 14, "e"), (15, 14, "e")],
    UP: [(14, y, "n") for y in range(2, 8)] + [(15, y, "m") for y in range(2, 8)] + [(14, 1, "e"), (15, 1, "e")],
    LEFT: [(x, 9, "n") for x in range(0, 4)] + [(x, 10, "m") for x in range(0, 4)] + [(0, 9, "e"), (0, 10, "e")],
}

# accessories (robot only): (x, y, role) per facing
ACCESSORIES = {
    "none": {},
    "scarf": {
        DOWN: [(x, 7, "c") for x in range(3, 13)] + [(12, 8, "c"), (12, 9, "c"), (13, 9, "c")],
        UP: [(x, 7, "c") for x in range(3, 13)] + [(7, 8, "c"), (8, 8, "c"), (7, 9, "c"), (8, 10, "c")],
        LEFT: [(x, 7, "c") for x in range(4, 13)] + [(12, 8, "c"), (13, 8, "c"), (13, 9, "c"), (14, 9, "c")],
    },
    "visor": {
        DOWN: [(x, 5, "c") for x in range(4, 12)] + [(5, 5, "e"), (6, 5, "e"), (9, 5, "e"), (10, 5, "e")]
              + [(3, 5, "k"), (12, 5, "k")],
        UP: [(x, 4, "c") for x in range(4, 12)],
        LEFT: [(x, 5, "c") for x in range(5, 12)] + [(3, 5, "c"), (4, 5, "c"), (5, 5, "e"), (6, 5, "e")],
    },
    "antennae": {
        d: [(4, 0, "y"), (4, 1, "k"), (4, 2, "k"), (11, 0, "y"), (11, 1, "k"), (11, 2, "k")]
        for d in (UP, DOWN)
    },
    "crest": {
        d: [(x, 2, "c") for x in range(4, 12)] + [(5, 1, "c"), (6, 1, "c"), (9, 1, "c"), (10, 1, "c"),
                                                  (5, 0, "e"), (10, 0, "e")]
        for d in (UP, DOWN)
    },
}
ACCESSORIES["antennae"][LEFT] = [(5, 0, "y"), (5, 1, "k"), (5, 2, "k"), (10, 0, "y"), (10, 1, "k"), (10, 2, "k")]
ACCESSORIES["crest"][LEFT] = [(x, 2, "c") for x in range(5, 12)] + [(6, 1, "c"), (7, 1, "c"), (10, 1, "c"),
                                                                      (6, 0, "e"), (10, 0, "e")]

# shield facing right (rotated for the other facings)
SHIELD_RIGHT = [BLANK] * 3 + [
    ".............kkk",
    "............kcck",
    "............kcak",
    "............kaak",
    "............kyyk",
    "............kyyk",
    "............kaak",
    "............kabk",
    "............kbbk",
    ".............kkk",
] + [BLANK] * 3


# ----------------------------------------------------------------------------------------------- helpers
def _check(grid, name):
    assert len(grid) == SIZE and all(len(r) == SIZE for r in grid), f"bad sprite grid {name}"
    return grid


def flip_h(grid):
    return [r[::-1] for r in grid]


def rot_cw(grid):
    return ["".join(grid[SIZE - 1 - x][y] for x in range(SIZE)) for y in range(SIZE)]


def rot_ccw(grid):
    return ["".join(grid[x][SIZE - 1 - y] for x in range(SIZE)) for y in range(SIZE)]


def orient(grid_right, facing):
    """A grid drawn for 'facing right' turned to `facing`."""
    return {RIGHT: lambda g: g, LEFT: flip_h, DOWN: rot_cw, UP: rot_ccw}[facing](grid_right)


def bob(top):
    """Idle frame 1: the upper body one pixel lower (the last torso row is dropped)."""
    return [BLANK] + top[:-1]


def _stamp(grid, pixels):
    g = [list(r) for r in grid]
    for x, y, role in pixels:
        if 0 <= x < SIZE and 0 <= y < SIZE:
            g[y][x] = role
    return ["".join(r) for r in g]


def _flip_pixels(pixels):
    return [(SIZE - 1 - x, y, r) for x, y, r in pixels]


def _facing_grids(table, facing):
    """(grid parts, mirrored?) for a facing from tables defined for UP / DOWN / LEFT."""
    return (table[LEFT], True) if facing == RIGHT else (table[facing], False)


def make_arc():
    """Slash arc in the tile in front of a fighter facing right (centre = the fighter, one tile to the left)."""
    rows = []
    for y in range(SIZE):
        row = ""
        for x in range(SIZE):
            dx, dy = x + 0.5 + 8.0, y + 0.5 - 8.0  # fighter centre sits 8 px left of this tile's left edge
            d, ang = math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))
            if abs(ang) <= 55 and 8.6 <= d <= 12.8:
                row += "c" if d >= 11.4 else ("w" if d >= 10.0 else "m")
            else:
                row += "."
        rows.append(row)
    return rows


ARC_RIGHT = make_arc()


class SpriteSet:
    """All frames of one character design in one look. `palette` maps palette names to RGB, `look` maps the
    roles a/b/c/e to palette names."""

    def __init__(self, kind, palette, look, accessory="none"):
        if kind not in ("knight", "robot"):
            raise ValueError(f"unknown sprite kind {kind!r}")
        if accessory not in ACCESSORIES:
            raise ValueError(f"unknown accessory {accessory!r}")
        self.kind, self.look = kind, dict(look)
        self.colors = {role: tuple(palette[name]) for role, name in {**FIXED, **look}.items()}
        top, legs, weapon = ((KNIGHT_TOP, KNIGHT_LEGS, KNIGHT_WEAPON) if kind == "knight"
                             else (ROBOT_TOP, ROBOT_LEGS, ROBOT_WEAPON))
        acc = ACCESSORIES[accessory] if kind == "robot" else {}
        self.frames = {}
        for f in (UP, DOWN, LEFT, RIGHT):
            tops, mirror = _facing_grids(top, f)
            legs_f, _ = _facing_grids(legs, f)
            weapon_px, _ = _facing_grids(weapon, f)
            acc_px = acc.get(LEFT if f == RIGHT else f, [])

            def build(top_rows, leg_key, armed=True):
                g = _stamp(top_rows + legs_f[leg_key], acc_px)
                g = _check(_stamp(g, weapon_px) if armed else g, f"{kind}/{f}")
                return flip_h(g) if mirror else g

            idle = build(tops, "stand")
            shield = orient(SHIELD_RIGHT, f)  # the shield replaces the weapon and sits on the front side
            blocked = ["".join(s if s != "." else b for s, b in zip(srow, brow))
                       for srow, brow in zip(shield, build(tops, "stand", armed=False))]
            grids = {"idle0": idle, "idle1": build(bob(tops), "stand"), "walk0": build(tops, "walk0"),
                     "walk1": build(tops, "walk1"), "attack": idle, "block": blocked}
            self.frames[f] = {k: self._surface(g) for k, g in grids.items()}
            self.frames[f]["ko"] = self._surface(self._ko(build(tops, "stand", armed=False), mirror))

    def _surface(self, grid):
        s = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
        for y, row in enumerate(grid):
            for x, role in enumerate(row):
                if role != ".":
                    s.set_at((x, y), self.colors[role])
        return s

    @staticmethod
    def _ko(grid, mirror):
        """Flat: the sprite lying on its side (head toward the front), squashed to every other row, with X eyes."""
        if mirror:  # undo the mirroring, lie down, mirror back so the head points the way the sprite faced
            grid = flip_h(grid)
        lying = rot_ccw(grid)
        flat = [BLANK] * 8 + [lying[y] for y in range(1, 15, 2)] + [BLANK]
        flat = _stamp(flat, [(3, 11, "k"), (4, 10, "w"), (4, 12, "w")])
        return flip_h(flat) if mirror else flat

    # ------------------------------------------------------------------ effects
    def arc(self, facing):
        """Slash arc as it appears in the front tile."""
        return self._surface(orient(ARC_RIGHT, facing))

    @staticmethod
    def silhouette(surface, color):
        return pygame.mask.from_surface(surface).to_surface(setcolor=tuple(color) + (255,),
                                                            unsetcolor=(0, 0, 0, 0))

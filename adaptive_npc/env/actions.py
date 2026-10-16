"""Action enum and direction constants."""
from enum import IntEnum


class Action(IntEnum):
    IDLE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    ATTACK = 5
    BLOCK = 6
    DODGE = 7


N_ACTIONS = len(Action)

# Facing index f in 0..3 = up, down, left, right. Move action a (1..4) sets facing a-1.
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
N_FACINGS = len(DIRS)

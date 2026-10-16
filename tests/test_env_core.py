import pytest

from adaptive_npc.config import load_config
from adaptive_npc.env.actions import Action
from adaptive_npc.env.core import ArenaGame
from adaptive_npc.env.observations import OBS_DIM, encode

IDLE, UP, DOWN, LEFT, RIGHT = Action.IDLE, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
ATTACK, BLOCK, DODGE = Action.ATTACK, Action.BLOCK, Action.DODGE


@pytest.fixture
def cfg():
    c = load_config("configs/env.yaml")
    c["random_start"] = False
    return c


@pytest.fixture
def g(cfg):
    """Players adjacent in the middle row: p0 at (3,4) facing right, p1 at (4,4) facing left."""
    g = ArenaGame(cfg)
    g.x, g.y = [3, 4], [4, 4]
    g.face = [3, 2]
    return g


def test_hit(g):
    r, done = g.step(ATTACK, IDLE)
    assert g.hp == [5, 4] and not done
    assert r == pytest.approx(0.1)
    assert g.cd[0] == 2


def test_attack_out_of_reach_misses(g):
    g.x[1] = 6
    g.step(ATTACK, IDLE)
    assert g.hp == [5, 5]


def test_simultaneous_attacks_trade(g):
    r, _ = g.step(ATTACK, ATTACK)
    assert g.hp == [4, 4]
    assert r == pytest.approx(0.0)


def test_block_parries_and_stuns_attacker(g):
    g.step(ATTACK, BLOCK)
    assert g.hp == [5, 5]
    assert g.parried[0] and g.stun[0] == 2
    assert g.sta[1] == 5  # block cost
    # stunned for 2 steps: its actions are ignored and the opponent hits freely
    g.cd[0] = 0
    g.step(ATTACK, ATTACK)
    assert g.hp == [4, 5]
    g.step(ATTACK, IDLE)
    assert g.stun[0] == 0
    g.step(IDLE, IDLE)  # step 4 after parry: stun is over
    assert g.stun[0] == 0


def test_stun_lasts_exactly_parry_stun_steps(g, cfg):
    g.step(ATTACK, BLOCK)
    g.stun[1] = 0
    moved = []
    for _ in range(cfg["block"]["parry_stun"]):
        g.step(DOWN, IDLE)
        moved.append(g.y[0] != 4)
    assert not any(moved)
    g.step(DOWN, IDLE)
    assert g.y[0] == 5


def test_block_only_protects_front(g):
    g.face[1] = 3  # p1 turned away from p0
    g.step(ATTACK, BLOCK)
    assert g.hp == [5, 4]
    assert not g.parried[0]


def test_dodge_iframes_and_cost(g):
    g.step(ATTACK, DODGE)  # dash is blocked by p0 standing in front, but i-frames hold
    assert g.hp == [5, 5]
    assert g.sta[1] == 3


def test_dodge_dashes_two_tiles(g):
    g.face[1] = 0  # up
    g.step(IDLE, DODGE)
    assert (g.x[1], g.y[1]) == (4, 2)


def test_dodge_stops_at_wall(g):
    g.y = [4, 1]
    g.face[1] = 0
    g.step(IDLE, DODGE)
    assert g.y[1] == 0


def test_dodge_needs_stamina(g):
    g.sta[1] = 2
    g.face[1] = 0
    g.step(ATTACK, DODGE)
    assert g.y[1] == 4 and g.sta[1] == 2  # fell back to idle
    assert g.hp[1] == 4  # no i-frames


def test_block_needs_stamina(g):
    g.sta[1] = 0
    g.step(ATTACK, BLOCK)
    assert g.hp == [5, 4] and not g.parried[0]


def test_stamina_regen(g, cfg):
    g.sta = [0, 0]
    g.step(IDLE, IDLE)
    assert g.sta == [0, 0]
    g.step(IDLE, IDLE)
    assert g.sta == [1, 1]
    g.sta = [cfg["stamina"]["max"]] * 2
    g.step(IDLE, IDLE)
    g.step(IDLE, IDLE)
    assert g.sta == [cfg["stamina"]["max"]] * 2


def test_attack_cooldown(g):
    g.step(ATTACK, IDLE)
    g.step(ATTACK, IDLE)
    g.step(ATTACK, IDLE)
    assert g.hp[1] == 4  # steps t+1, t+2 blocked by cooldown
    g.step(ATTACK, IDLE)
    assert g.hp[1] == 3


def test_move_sets_facing_and_blocked_by_opponent(g):
    g.step(RIGHT, IDLE)  # p1 occupies the target tile
    assert (g.x[0], g.y[0]) == (3, 4) and g.face[0] == 3
    g.step(UP, DOWN)
    assert (g.y[0], g.y[1]) == (3, 5) and g.face == [0, 1]


def test_moves_into_same_tile_cancel(g):
    g.x, g.y = [3, 5], [4, 4]
    g.step(RIGHT, LEFT)
    assert g.x == [3, 5]


def test_walls(g):
    g.x, g.y = [0, 8], [0, 8]
    g.step(LEFT, RIGHT)
    g.step(UP, DOWN)
    assert (g.x[0], g.y[0], g.x[1], g.y[1]) == (0, 0, 8, 8)


def test_kill_ends_round(g):
    g.hp[1] = 1
    r, done = g.step(ATTACK, IDLE)
    assert done and g.winner == 0
    assert r == pytest.approx(0.1 + 1.0)


def test_double_ko_is_draw(g):
    g.hp = [1, 1]
    r, done = g.step(ATTACK, ATTACK)
    assert done and g.winner == -1
    assert r == pytest.approx(0.0)


def test_timeout_higher_hp_wins(cfg):
    cfg["max_steps"] = 3
    g = ArenaGame(cfg)
    g.hp = [5, 4]
    for i in range(3):
        assert not g.done
        r, done = g.step(IDLE, IDLE)
    assert done and g.winner == 0 and r == pytest.approx(1.0)


def test_timeout_equal_hp_is_draw(cfg):
    cfg["max_steps"] = 2
    g = ArenaGame(cfg)
    g.step(IDLE, IDLE)
    r, done = g.step(IDLE, IDLE)
    assert done and g.winner == -1 and r == 0.0


def test_loss_reward(g):
    g.hp[0] = 1
    r, done = g.step(IDLE, ATTACK)
    assert done and g.winner == 1
    assert r == pytest.approx(-0.1 - 1.0)


def test_observation_is_mirrored(g):
    o0, o1 = encode(g, 0), encode(g, 1)
    assert o0.shape == (OBS_DIM,)
    # relative position flips sign, own/opp blocks swap
    assert o0[2] == pytest.approx(-o1[2]) and o0[3] == pytest.approx(-o1[3])
    assert (o0[4:8] == o1[8:12]).all() and (o0[8:12] == o1[4:8]).all()


def test_random_start_is_seeded_and_valid(cfg):
    cfg["random_start"] = True
    a, b = ArenaGame(cfg), ArenaGame(cfg)
    for _ in range(20):
        a.reset()
        b.reset()
        assert (a.x, a.y) == (b.x, b.y)
        assert a.x[0] < a.x[1]

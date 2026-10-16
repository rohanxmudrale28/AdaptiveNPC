"""Observation encoding. The same function serves both views (who=0 NPC, who=1 player),
so a player-view observation is the exact mirror of the NPC-view one."""
import numpy as np

from adaptive_npc.env.actions import N_FACINGS

# own pos (2) | rel pos to opponent (2) | own facing (4) | opp facing (4) | hp (2)
# | stamina (2) | attack cooldown (2) | stun (2) | time fraction (1)
OBS_DIM = 2 + 2 + N_FACINGS + N_FACINGS + 2 + 2 + 2 + 2 + 1


def encode(g, who):
    """Return a float32 vector of length OBS_DIM from player `who`'s point of view."""
    o = 1 - who
    span = max(g.size - 1, 1)
    cd_max = max(g.atk_cooldown, 1)
    stun_max = max(g.parry_stun, 1)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = g.x[who] / span
    obs[1] = g.y[who] / span
    obs[2] = (g.x[o] - g.x[who]) / span
    obs[3] = (g.y[o] - g.y[who]) / span
    obs[4 + g.face[who]] = 1.0
    obs[8 + g.face[o]] = 1.0
    obs[12] = g.hp[who] / g.hp_max
    obs[13] = g.hp[o] / g.hp_max
    obs[14] = g.sta[who] / g.sta_max
    obs[15] = g.sta[o] / g.sta_max
    obs[16] = g.cd[who] / cd_max
    obs[17] = g.cd[o] / cd_max
    obs[18] = g.stun[who] / stun_max
    obs[19] = g.stun[o] / stun_max
    obs[20] = g.t / g.max_steps
    return obs

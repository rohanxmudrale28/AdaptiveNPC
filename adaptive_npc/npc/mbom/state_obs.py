"""Batched observation encoding from raw integer states (the env model's 15-int state vector).

Imagined states from Γ are raw integers; the policy and the IOPs need the observation vector. `encode_raw` is a
batched torch port of `observations.encode` and gives the same float32 values (computed in float64, then cast,
like the numpy version).
"""
import torch

from adaptive_npc.env.actions import N_FACINGS
from adaptive_npc.env.observations import OBS_DIM
from adaptive_npc.models.env_model import CD0, F0, HP0, STA0, STUN0, T, X0, Y0, EnvSpec


def encode_raw(raw, who, spec):
    """raw: int tensor / array [N, 15]; who: 0 (NPC view) or 1 (player view); spec: EnvSpec. -> float32 [N, OBS_DIM]."""
    if not isinstance(spec, EnvSpec):
        spec = EnvSpec(spec)
    s = torch.as_tensor(raw).long()
    n, o = s.shape[0], 1 - who
    span = float(max(spec.size - 1, 1))
    cd_max, stun_max = float(max(spec.cd_max, 1)), float(max(spec.stun_max, 1))
    sd = s.double()
    xw, yw, xo, yo = sd[:, X0 + 2 * who], sd[:, Y0 + 2 * who], sd[:, X0 + 2 * o], sd[:, Y0 + 2 * o]
    obs = torch.zeros(n, OBS_DIM, dtype=torch.float64)
    obs[:, 0] = xw / span
    obs[:, 1] = yw / span
    obs[:, 2] = (xo - xw) / span
    obs[:, 3] = (yo - yw) / span
    rows = torch.arange(n)
    obs[rows, 4 + s[:, F0 + who]] = 1.0
    obs[rows, 4 + N_FACINGS + s[:, F0 + o]] = 1.0
    obs[:, 12] = sd[:, HP0 + who] / spec.hp_max
    obs[:, 13] = sd[:, HP0 + o] / spec.hp_max
    obs[:, 14] = sd[:, STA0 + who] / spec.sta_max
    obs[:, 15] = sd[:, STA0 + o] / spec.sta_max
    obs[:, 16] = sd[:, CD0 + who] / cd_max
    obs[:, 17] = sd[:, CD0 + o] / cd_max
    obs[:, 18] = sd[:, STUN0 + who] / stun_max
    obs[:, 19] = sd[:, STUN0 + o] / stun_max
    obs[:, 20] = sd[:, T] / spec.max_steps
    return obs.float()

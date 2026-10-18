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


def decode_obs(obs, who, spec):
    """Inverse of encode_raw: observation(s) [N, OBS_DIM] (or [OBS_DIM]) from `who`'s view -> raw long [N, 15].
    Every field is a scaled integer or a one-hot, so the state is recovered exactly (rounding)."""
    if not isinstance(spec, EnvSpec):
        spec = EnvSpec(spec)
    o = torch.as_tensor(obs, dtype=torch.float64).reshape(-1, OBS_DIM)
    span = float(max(spec.size - 1, 1))
    cd_max, stun_max = float(max(spec.cd_max, 1)), float(max(spec.stun_max, 1))
    r = lambda x: torch.round(x).long()  # noqa: E731
    w, op = who, 1 - who
    raw = torch.zeros(o.shape[0], 15, dtype=torch.long)
    raw[:, X0 + 2 * w], raw[:, Y0 + 2 * w] = r(o[:, 0] * span), r(o[:, 1] * span)
    raw[:, X0 + 2 * op] = raw[:, X0 + 2 * w] + r(o[:, 2] * span)
    raw[:, Y0 + 2 * op] = raw[:, Y0 + 2 * w] + r(o[:, 3] * span)
    raw[:, F0 + w], raw[:, F0 + op] = o[:, 4:8].argmax(1), o[:, 8:12].argmax(1)
    raw[:, HP0 + w], raw[:, HP0 + op] = r(o[:, 12] * spec.hp_max), r(o[:, 13] * spec.hp_max)
    raw[:, STA0 + w], raw[:, STA0 + op] = r(o[:, 14] * spec.sta_max), r(o[:, 15] * spec.sta_max)
    raw[:, CD0 + w], raw[:, CD0 + op] = r(o[:, 16] * cd_max), r(o[:, 17] * cd_max)
    raw[:, STUN0 + w], raw[:, STUN0 + op] = r(o[:, 18] * stun_max), r(o[:, 19] * stun_max)
    raw[:, T] = r(o[:, 20] * spec.max_steps)
    return raw

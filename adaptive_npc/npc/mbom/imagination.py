"""Imagined opponent best responses (MBOM eq. 5), batched in torch.

For each real state s (where the opponent acted) and each candidate opponent action c:
  t = 0       opponent plays c, the NPC plays a ~ π(·| obs(s), IOP_{m-1}(window))
  t = 1..k    opponent plays a uniformly random continuation, the NPC keeps acting with π∘IOP_{m-1}
  return      G = Σ_{t=0..k} γ^t rᵒ_t + γ^{k+1} Vᵒ(s_{k+1}),  rᵒ = -r and Vᵒ = -V (zero-sum)
Rewards after a terminal step and the bootstrap of a finished round are masked out (Γ keeps stepping).
Each imagined step extends the IOP window with the imagined (obs, NPC action, opponent action) row.

Common random numbers: the opponent continuations (opp_samples per state) and the NPC's sampling noise
(npc_samples uniforms per state and step, inverse-CDF sampling) are drawn once per state and shared by all 8
candidates, so candidates are compared on the same randomness.
score_agg 'max':  Q(c) = max over opponent continuations of the mean over NPC samples (the NPC's randomness is
                  averaged, the opponent picks its best continuation)
score_agg 'mean': Q(c) = mean over both
imagined_npc 'greedy' uses the NPC's argmax action (npc_samples is then 1).
"""
import torch
import torch.nn.functional as F

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.npc.mbom.state_obs import encode_raw

SCORE_AGGS = ("max", "mean")
NPC_MODES = ("stochastic", "greedy")


def policy_fns(policy):
    """SB3 ActorCriticPolicy -> (npc_probs(x) [N, 8], npc_value(x) [N]) on policy inputs x [N, obs + 8]."""
    @torch.no_grad()
    def probs(x):
        return policy.get_distribution(x).distribution.probs

    @torch.no_grad()
    def value(x):
        return policy.predict_values(x)[:, 0]
    return probs, value


def _pick(probs, u, greedy):
    if greedy:
        return probs.argmax(1)
    cdf = probs.cumsum(1)
    return (cdf < u[:, None] * cdf[:, -1:]).sum(1).clamp(max=N_ACTIONS - 1)


def _row(obs, a_npc, a_opp):
    return torch.cat([obs, F.one_hot(a_npc, N_ACTIONS).float(), F.one_hot(a_opp, N_ACTIONS).float()], 1)


@torch.no_grad()
def imagined_returns(env_model, spec, npc_probs, npc_value, iop_prev, raw0, win0, gen, k=2, opp_samples=16,
                     npc_samples=1, gamma=0.99, imagined_npc="stochastic", chunk_rows=32768):
    """raw0: long [B, 15] real states; win0: float [B, H, FEAT] IOP windows at those states (last row = s).
    iop_prev(wins [N, H, FEAT]) -> probs [N, 8]. Returns G: float [B, 8 candidates, opp_samples, npc_samples]."""
    if imagined_npc not in NPC_MODES:
        raise ValueError(f"imagined_npc must be one of {NPC_MODES}, got {imagined_npc!r}")
    greedy = imagined_npc == "greedy"
    S = 1 if greedy else int(npc_samples)
    B, C, J = raw0.shape[0], N_ACTIONS, int(opp_samples)
    # all randomness drawn up front, so results do not depend on chunking
    u = torch.rand(B, S, k + 1, generator=gen)
    opp = torch.randint(N_ACTIONS, (B, J, max(k, 1)), generator=gen)
    per_state = C * J * S
    bs = max(1, chunk_rows // per_state)
    out = []
    for b0 in range(0, B, bs):
        sl = slice(b0, min(b0 + bs, B))
        out.append(_chunk(env_model, spec, npc_probs, npc_value, iop_prev, raw0[sl].long(), win0[sl], u[sl],
                          opp[sl], k, gamma, greedy))
    return torch.cat(out).reshape(B, C, J, S)


def _chunk(env_model, spec, npc_probs, npc_value, iop_prev, raw0, win0, u, opp, k, gamma, greedy):
    b, S, J, C = raw0.shape[0], u.shape[1], opp.shape[1], N_ACTIONS
    R = b * C * J * S

    def expand(x, dims):  # x [b, *] laid out over the (b, C, J, S) row grid; dims = which of C/J/S x varies on
        shape = [b] + [n if d in dims else 1 for d, n in zip("CJS", (C, J, S))]
        return x.reshape(shape).expand(b, C, J, S).reshape(R)

    # t = 0: the state, the IOP window and the NPC's sampled action are the same for every candidate
    obs = encode_raw(raw0, 0, spec)
    p = iop_prev(win0)
    a0 = torch.stack([_pick(npc_probs(torch.cat([obs, p], 1)), u[:, s, 0], greedy) for s in range(S)], 1)
    state = raw0[:, None].expand(b, C * J * S, raw0.shape[1]).reshape(R, -1)
    win = win0[:, None].expand(b, C * J * S, *win0.shape[1:]).reshape(R, *win0.shape[1:])
    a_npc = expand(a0, "S")
    a_opp = expand(torch.arange(C)[None].expand(b, C), "C")
    state, r, done = env_model.step_t(state, a_npc, a_opp)
    G = -r.double()
    alive = ~done
    for t in range(1, k + 1):
        obs = encode_raw(state, 0, spec)
        win = torch.cat([win[:, 1:], _row(obs, a_npc, a_opp)[:, None]], 1)
        x = torch.cat([obs, iop_prev(win)], 1)
        a_npc = _pick(npc_probs(x), expand(u[:, :, t], "S"), greedy)
        a_opp = expand(opp[:, :, t - 1], "J")
        state, r, d = env_model.step_t(state, a_npc, a_opp)
        G += gamma ** t * alive.double() * -r.double()
        alive = alive & ~d
    obs = encode_raw(state, 0, spec)
    win = torch.cat([win[:, 1:], _row(obs, a_npc, a_opp)[:, None]], 1)
    v = npc_value(torch.cat([obs, iop_prev(win)], 1))
    G += gamma ** (k + 1) * alive.double() * -v.double()
    return G.float().reshape(b, C, J, S)


def aggregate_q(G, score_agg="max"):
    """G [B, 8, J, S] -> Q [B, 8]."""
    if score_agg == "max":
        return G.mean(3).max(2).values
    if score_agg == "mean":
        return G.mean((2, 3))
    raise ValueError(f"score_agg must be one of {SCORE_AGGS}, got {score_agg!r}")


def br_targets(Q, tau):
    """Soft best-response targets softmax(Q / tau) [B, 8]."""
    return F.softmax(Q / float(tau), dim=1)

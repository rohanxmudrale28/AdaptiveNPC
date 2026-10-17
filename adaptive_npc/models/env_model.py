"""Env model Γ: (state, NPC action, opponent action) -> (next state, reward). MBOM eq. 2, MLP [256, 256].

The state is the full game state as 15 integers (NPC = player 0, opponent = player 1; absolute coordinates):
    x0 y0 x1 y1 | face0 face1 | hp0 hp1 | sta0 sta1 | cd0 cd1 | stun0 stun1 | t
The game is Markov in this vector (all randomness is in reset), so Γ can in principle be exact.

Discrete handling (everything in the state is a small integer, so everything is a classification):
  * positions, hp, stamina, cooldown, stun: one classification head per channel over the integer delta in
    [-m, m] (m = largest |delta| of the channel: dodge distance, damage, dodge cost, cooldown, stun); argmax,
    then clipped to the legal range
  * facing: two 4-way heads (next facing); t: not learned, always t + 1
  * reward: the net regresses the shaping part (damage dealt / taken, snapped to the reward quantum); the terminal
    win / loss / draw bonus is computed from the PREDICTED next state with the configured reward values, because it
    is a deterministic function of hp and t after the step (a rare +-1 spike is hard for an MSE head)
  * inputs: one-hot positions and relative position (wall and collision edges), facing one-hot, t mod regen_every
    and a "last step" flag (stamina regen / timeout)
The snapped integer state is what feeds back in rollouts, so errors do not blur into non-integer states.
`TrueSimulator` has the same step() interface and is the debug-only `oracle_model` (never used in reported results).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_npc.env.actions import N_ACTIONS, N_FACINGS
from adaptive_npc.env.core import ArenaGame

X0, Y0, X1, Y1, F0, F1, HP0, HP1, STA0, STA1, CD0, CD1, STUN0, STUN1, T = range(15)
STATE_DIM = 15
CHANNELS = ("x_npc", "y_npc", "x_opp", "y_opp", "face_npc", "face_opp", "hp_npc", "hp_opp",
            "sta_npc", "sta_opp", "cd_npc", "cd_opp", "stun_npc", "stun_opp")  # state channels except t
CHANNEL_IDX = (X0, Y0, X1, Y1, F0, F1, HP0, HP1, STA0, STA1, CD0, CD1, STUN0, STUN1)
REG = (X0, Y0, X1, Y1, HP0, HP1, STA0, STA1, CD0, CD1, STUN0, STUN1)  # regressed as deltas
FACE = (F0, F1)
N_REG = len(REG)


def raw_state(g):
    """Integer state vector of an ArenaGame (health, timers etc. as the rules see them)."""
    return np.array([g.x[0], g.y[0], g.x[1], g.y[1], g.face[0], g.face[1], g.hp[0], g.hp[1], g.sta[0], g.sta[1],
                     g.cd[0], g.cd[1], g.stun[0], g.stun[1], g.t], dtype=np.int16)


class EnvSpec:
    """Game constants the model needs, read from configs/env.yaml (no hardcoded rules)."""

    def __init__(self, env_cfg):
        self.cfg = env_cfg
        self.size, self.max_steps = int(env_cfg["arena_size"]), int(env_cfg["max_steps"])
        self.hp_max, self.sta_max = int(env_cfg["hp"]), int(env_cfg["stamina"]["max"])
        self.regen_every = int(env_cfg["stamina"]["regen_every"])
        self.cd_max, self.stun_max = int(env_cfg["attack"]["cooldown"]), int(env_cfg["block"]["parry_stun"])
        dodge_dist, dodge_cost = int(env_cfg["dodge"]["distance"]), int(env_cfg["dodge"]["stamina_cost"])
        dmg = int(env_cfg["attack"]["damage"])
        # delta scales (about the largest |delta| of each channel), so regression targets are ~unit range
        self.scales = np.array([dodge_dist] * 4 + [dmg] * 2 + [dodge_cost] * 2 + [self.cd_max] * 2
                               + [self.stun_max] * 2, dtype=np.float32)
        self.delta_m = self.scales.astype(int)  # delta classes cover [-m, m]
        self.head_sizes = [2 * int(m) + 1 for m in self.delta_m] + [N_FACINGS, N_FACINGS]  # 12 deltas, 2 facings
        self.head_offsets = np.concatenate([[0], np.cumsum(self.head_sizes)]).astype(int)
        self.n_out = int(self.head_offsets[-1]) + 1  # + shaping reward
        rw = env_cfg["reward"]
        self.r_win, self.r_loss, self.r_draw = float(rw["win"]), float(rw["loss"]), float(rw["draw"])
        # shaping rewards are multiples of one quantum when the dealt / taken weights match in size
        self.r_quantum = abs(float(rw["damage_dealt"])) if abs(rw["damage_dealt"]) == abs(rw["damage_taken"]) else 0.0
        self.r_scale = self.r_quantum if self.r_quantum > 0 else 1.0  # the reward head works in units of this
        self.lo = np.zeros(STATE_DIM, dtype=np.int16)
        self.hi = np.array([self.size - 1] * 4 + [N_FACINGS - 1] * 2 + [self.hp_max] * 2 + [self.sta_max] * 2
                           + [self.cd_max] * 2 + [self.stun_max] * 2 + [self.max_steps], dtype=np.int16)
        rel = 2 * self.size - 1
        self.in_dim = 4 + 4 * self.size + 2 * rel + 2 * N_FACINGS + 8 + 1 + self.regen_every + 1 + 2 * N_ACTIONS

    def done(self, raw):
        return (raw[:, HP0] <= 0) | (raw[:, HP1] <= 0) | (raw[:, T] >= self.max_steps)

    def terminal_bonus(self, raw):
        """Win / loss / draw reward at the end of a round, 0 elsewhere (same rule as ArenaGame.step)."""
        bonus = np.where(raw[:, HP0] > raw[:, HP1], self.r_win,
                         np.where(raw[:, HP1] > raw[:, HP0], self.r_loss, self.r_draw))
        return np.where(self.done(raw), bonus, 0.0).astype(np.float32)


def features(spec, raw, a_npc, a_opp):
    """(N, 15) int raw states + (N,) actions -> (N, in_dim) float32 network input."""
    raw = torch.as_tensor(raw).long()
    a0, a1 = torch.as_tensor(a_npc).long(), torch.as_tensor(a_opp).long()
    t = raw[:, T]
    n = spec.size
    parts = [raw[:, [X0, Y0, X1, Y1]].float() / max(n - 1, 1)]
    parts += [F.one_hot(raw[:, c], n).float() for c in (X0, Y0, X1, Y1)]
    parts += [F.one_hot(raw[:, X1] - raw[:, X0] + n - 1, 2 * n - 1).float(),
              F.one_hot(raw[:, Y1] - raw[:, Y0] + n - 1, 2 * n - 1).float()]
    parts += [F.one_hot(raw[:, F0], N_FACINGS).float(), F.one_hot(raw[:, F1], N_FACINGS).float(),
             raw[:, [HP0, HP1]].float() / spec.hp_max, raw[:, [STA0, STA1]].float() / spec.sta_max,
             raw[:, [CD0, CD1]].float() / max(spec.cd_max, 1), raw[:, [STUN0, STUN1]].float() / max(spec.stun_max, 1),
             (t.float() / spec.max_steps)[:, None], F.one_hot(t % spec.regen_every, spec.regen_every).float(),
             ((t + 1 >= spec.max_steps).float())[:, None],
             F.one_hot(a0, N_ACTIONS).float(), F.one_hot(a1, N_ACTIONS).float()]
    return torch.cat(parts, dim=1)


def targets(spec, raw, nxt, reward):
    """Training targets: class indices (N, 14) = 12 delta classes (delta + m) then 2 facings, and the shaping reward
    (reward minus the terminal bonus of the true next state) (N,)."""
    raw, nxt = np.asarray(raw), np.asarray(nxt)
    delta = nxt[:, REG].astype(int) - raw[:, REG].astype(int) + spec.delta_m
    cls = np.concatenate([np.clip(delta, 0, 2 * spec.delta_m), nxt[:, list(FACE)].astype(int)], axis=1)
    shaping = (np.asarray(reward, dtype=np.float32) - spec.terminal_bonus(nxt)) / spec.r_scale
    return torch.as_tensor(cls).long(), torch.as_tensor(shaping, dtype=torch.float32)


def loss_fn(spec, out, cls_t, r_t, reward_weight=1.0):
    """Sum over the 14 heads of cross-entropy (each head's batch mean, so rare events are not diluted) + MSE on the
    shaping reward."""
    ce = sum(F.cross_entropy(out[:, spec.head_offsets[i]:spec.head_offsets[i + 1]], cls_t[:, i])
             for i in range(len(spec.head_sizes)))
    mse_r = ((out[:, -1] - r_t) ** 2).mean()
    return ce + reward_weight * mse_r, {"ce": ce.item(), "mse_r": mse_r.item()}


class EnvModel(nn.Module):
    def __init__(self, spec, hidden=(256, 256)):
        super().__init__()
        self.spec, self.hidden = spec, [int(h) for h in hidden]
        layers, d = [], spec.in_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, spec.n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def step(self, raw, a_npc, a_opp, chunk=65536):
        """Batched one step on integer states. Returns (next_raw int16 (N, 15), reward float32 (N,), done bool (N,))."""
        raw = np.asarray(raw, dtype=np.int16)
        was_training = self.training
        self.eval()
        nxt, rew, done = self.step_t(torch.from_numpy(raw.astype(np.int64)), torch.as_tensor(np.asarray(a_npc)),
                                     torch.as_tensor(np.asarray(a_opp)), chunk)
        self.train(was_training)
        return nxt.numpy().astype(np.int16), rew.numpy(), done.numpy()

    @torch.no_grad()
    def step_t(self, raw, a_npc, a_opp, chunk=65536):
        """step() in torch, no numpy round trip (for batched imagination). raw: long tensor (N, 15); actions: (N,).
        Returns (next_raw long (N, 15), reward float32 (N,), done bool (N,)). The net has no dropout / batch norm,
        so train / eval mode does not matter here."""
        raw = raw.long()
        sp, off = self.spec, self.spec.head_offsets
        a_npc, a_opp = a_npc.long(), a_opp.long()
        outs = [self(features(sp, raw[i:i + chunk], a_npc[i:i + chunk], a_opp[i:i + chunk]))
                for i in range(0, len(raw), chunk)]
        out = torch.cat(outs) if outs else torch.zeros((0, sp.n_out))
        cls = torch.stack([out[:, off[i]:off[i + 1]].argmax(1) for i in range(len(sp.head_sizes))], 1)
        nxt = raw.clone()
        nxt[:, list(REG)] += cls[:, :N_REG] - torch.as_tensor(sp.delta_m, dtype=torch.long)
        nxt[:, F0], nxt[:, F1] = cls[:, N_REG], cls[:, N_REG + 1]
        nxt[:, T] = raw[:, T] + 1
        nxt = torch.maximum(torch.minimum(nxt, torch.as_tensor(sp.hi, dtype=torch.long)),
                            torch.as_tensor(sp.lo, dtype=torch.long))
        shaping = out[:, -1]
        if sp.r_quantum > 0:
            shaping = torch.round(shaping)
        shaping = shaping * sp.r_scale
        hp0, hp1 = nxt[:, HP0], nxt[:, HP1]
        done = (hp0 <= 0) | (hp1 <= 0) | (nxt[:, T] >= sp.max_steps)
        bonus = torch.where(hp0 > hp1, sp.r_win, torch.where(hp1 > hp0, sp.r_loss, sp.r_draw)).double()
        bonus = torch.where(done, bonus, torch.zeros_like(bonus))
        return nxt, (shaping.double() + bonus).float(), done


class TrueSimulator:
    """The real rules behind Γ's interface. DEBUG ONLY (`oracle_model`): upper bound for imagination, never reported."""

    def __init__(self, env_cfg):
        self.spec = EnvSpec(env_cfg)
        self.game = ArenaGame(env_cfg)

    def step(self, raw, a_npc, a_opp):
        raw = np.asarray(raw, dtype=np.int16)
        g = self.game
        nxt, rew, done = np.zeros_like(raw), np.zeros(len(raw), np.float32), np.zeros(len(raw), bool)
        for i, s in enumerate(raw):
            g.x, g.y = [int(s[X0]), int(s[X1])], [int(s[Y0]), int(s[Y1])]
            g.face, g.hp = [int(s[F0]), int(s[F1])], [int(s[HP0]), int(s[HP1])]
            g.sta, g.cd, g.stun = [int(s[STA0]), int(s[STA1])], [int(s[CD0]), int(s[CD1])], [int(s[STUN0]), int(s[STUN1])]
            g.t, g.done, g.winner = int(s[T]), False, None
            r, d = g.step(int(a_npc[i]), int(a_opp[i]))
            nxt[i], rew[i], done[i] = raw_state(g), r, d
        return nxt, rew, done

    def step_t(self, raw, a_npc, a_opp):
        """Torch-tensor interface of step() (same as EnvModel.step_t); still a Python loop over rows."""
        nxt, rew, done = self.step(raw.numpy(), a_npc.numpy(), a_opp.numpy())
        return torch.from_numpy(nxt.astype(np.int64)), torch.from_numpy(rew), torch.from_numpy(done)


def rollout(model, raw, a_npc_seq, a_opp_seq):
    """k-step open-loop rollout from (N, 15) states with (N, k) action sequences.
    Returns states (N, k + 1, 15), rewards (N, k) and done flags (N, k). Works for Γ and TrueSimulator."""
    a_npc_seq, a_opp_seq = np.asarray(a_npc_seq), np.asarray(a_opp_seq)
    states, rewards, dones = [np.asarray(raw, dtype=np.int16)], [], []
    for k in range(a_npc_seq.shape[1]):
        nxt, r, d = model.step(states[-1], a_npc_seq[:, k], a_opp_seq[:, k])
        states.append(nxt), rewards.append(r), dones.append(d)
    return np.stack(states, 1), np.stack(rewards, 1), np.stack(dones, 1)


def save_env_model(path, model, env_cfg, extra=None):
    torch.save({"state_dict": model.state_dict(), "hidden": model.hidden, "env_cfg": env_cfg, **(extra or {})}, path)


def load_env_model(path, oracle_model=False, env_cfg=None):
    """Load Γ from a checkpoint. oracle_model=True returns the TrueSimulator instead (debug only; needs env_cfg or the
    checkpoint's saved env config)."""
    if oracle_model and env_cfg is not None:
        return TrueSimulator(env_cfg)  # no checkpoint needed
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = env_cfg if env_cfg is not None else ck["env_cfg"]
    if oracle_model:
        return TrueSimulator(cfg)
    model = EnvModel(EnvSpec(cfg), ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    return model.eval()

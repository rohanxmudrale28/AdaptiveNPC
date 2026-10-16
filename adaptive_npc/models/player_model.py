"""GRU next-action predictor for the opponent (player) plus its online-finetuning wrapper.

Input per step i of a round: [obs_i (NPC view), onehot(npc action at i-1), onehot(player action at i-1)]
(previous-action slots are zero at round start). A window of the last H steps is fed to the GRU and the
last output predicts the player's action at step i. Windows are left-padded with all-zero rows
(a real obs is never all zeros: the facing one-hot always sums to 1).
"""
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.env.observations import OBS_DIM

FEAT_DIM = OBS_DIM + 2 * N_ACTIONS


class PlayerPredictor(nn.Module):
    def __init__(self, hidden=64, layers=1):
        super().__init__()
        self.gru = nn.GRU(FEAT_DIM, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, N_ACTIONS)

    def forward(self, x):
        """x: [B, H, FEAT_DIM] -> logits [B, N_ACTIONS]."""
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def build_model(model_cfg):
    return PlayerPredictor(int(model_cfg["hidden"]), int(model_cfg["layers"]))


def step_features(obs, prev_npc, prev_pl):
    """Feature row for one step. prev_* = None at round start."""
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    f[:OBS_DIM] = obs
    if prev_npc is not None:
        f[OBS_DIM + prev_npc] = 1.0
        f[OBS_DIM + N_ACTIONS + prev_pl] = 1.0
    return f


class WindowData:
    """Flat per-step arrays -> windows built lazily per batch (avoids storing N x H x FEAT_DIM)."""

    def __init__(self, obs, a_npc, a_pl, round_start, window):
        n = len(obs)
        prev_npc = np.zeros((n, N_ACTIONS), dtype=np.float32)
        prev_pl = np.zeros((n, N_ACTIONS), dtype=np.float32)
        prev_npc[1:] = np.eye(N_ACTIONS, dtype=np.float32)[a_npc[:-1]]
        prev_pl[1:] = np.eye(N_ACTIONS, dtype=np.float32)[a_pl[:-1]]
        prev_npc[round_start] = 0.0
        prev_pl[round_start] = 0.0
        # extra all-zero row at index n is the padding sentinel
        self.X = np.concatenate([np.concatenate([obs, prev_npc, prev_pl], axis=1),
                                 np.zeros((1, FEAT_DIM), dtype=np.float32)]).astype(np.float32)
        first = np.maximum.accumulate(np.where(round_start, np.arange(n), 0))
        idx = np.arange(n)[:, None] + np.arange(-window + 1, 1)[None, :]
        self.idx = np.where(idx >= first[:, None], idx, n)
        self.y = a_pl.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def batch(self, ids):
        return torch.from_numpy(self.X[self.idx[ids]]), torch.from_numpy(self.y[ids])


class OnlinePredictor:
    """Stateful predictor used inside a match. predict(obs) before the step, observe(a_npc, a_pl) after.
    With ft_cfg=None the weights stay frozen; otherwise every `every` steps it takes `grad_steps`
    Adam steps on the last `buffer` (window, action) pairs. Works on a private copy of the model."""

    def __init__(self, model, window, ft_cfg=None):
        self.model = copy.deepcopy(model).eval()
        self.window = int(window)
        self.ft = ft_cfg
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(ft_cfg["lr"])) if ft_cfg else None
        self.buffer = deque(maxlen=int(ft_cfg["buffer"])) if ft_cfg else None
        self.steps = 0
        self.prev = (None, None)
        self.reset_round()

    def reset_round(self):
        self._rows = []
        self.prev = (None, None)
        self._pending = None

    @torch.no_grad()
    def predict(self, obs):
        self._rows.append(step_features(obs, *self.prev))
        win = np.zeros((self.window, FEAT_DIM), dtype=np.float32)
        recent = self._rows[-self.window:]
        win[self.window - len(recent):] = recent
        self._pending = torch.from_numpy(win)
        logits = self.model(self._pending[None])
        return F.softmax(logits, dim=-1)[0].numpy()

    def observe(self, a_npc, a_pl):
        self.prev = (int(a_npc), int(a_pl))
        self.steps += 1
        if self.ft is None:
            return
        self.buffer.append((self._pending, int(a_pl)))
        if self.steps % int(self.ft["every"]) == 0 and len(self.buffer) >= int(self.ft["min_buffer"]):
            self._finetune()

    def _finetune(self):
        x = torch.stack([w for w, _ in self.buffer])
        y = torch.tensor([a for _, a in self.buffer])
        self.model.train()
        for _ in range(int(self.ft["grad_steps"])):
            loss = F.cross_entropy(self.model(x), y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.model.eval()


class MajorityBaseline:
    """Causal most-frequent-action baseline: Laplace-smoothed counts over the match so far."""

    def __init__(self):
        self.counts = np.ones(N_ACTIONS)

    def reset_round(self):
        pass

    def predict(self, obs):
        return self.counts / self.counts.sum()

    def observe(self, a_npc, a_pl):
        self.counts[a_pl] += 1


class RepeatLastBaseline:
    """Repeat the player's previous action (carried across rounds); eps-smoothed so KL is finite."""

    def __init__(self, eps):
        self.eps = float(eps)
        self.last = None

    def reset_round(self):
        pass

    def predict(self, obs):
        p = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
        if self.last is not None:
            p = np.full(N_ACTIONS, self.eps / (N_ACTIONS - 1))
            p[self.last] = 1.0 - self.eps
        return p

    def observe(self, a_npc, a_pl):
        self.last = int(a_pl)


def save_predictor(path, model, model_cfg):
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg}, path)


def load_predictor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model_cfg"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["model_cfg"]

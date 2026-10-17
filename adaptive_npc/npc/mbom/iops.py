"""Imagined opponent policies φ0..φ_{M-1}: GRU predictors with the PlayerPredictor architecture.

φ0 starts as the pretrained predictor and is finetuned online on the real opponent actions. φ_m (m >= 1) starts each
match as a copy of φ0 and is finetuned incrementally (it persists across updates) on soft best-response targets
computed by imagination against π∘φ_{m-1}. All levels share the real-history buffer: the last `buffer` entries of
(IOP window, raw state, opponent action).
"""
import copy
from collections import deque

import torch
import torch.nn.functional as F


class IOPSet:
    def __init__(self, model, n_levels, lr, grad_steps, buffer):
        self.levels = [copy.deepcopy(model).eval() for _ in range(int(n_levels))]
        self.opts = [torch.optim.Adam(m.parameters(), lr=float(lr)) for m in self.levels]
        self.grad_steps = int(grad_steps)
        self.buffer = deque(maxlen=int(buffer))

    @property
    def M(self):
        return len(self.levels)

    @torch.no_grad()
    def probs(self, m, wins):
        """wins [N, H, FEAT] -> level m's probabilities [N, 8]."""
        return F.softmax(self.levels[m](wins), dim=-1)

    @torch.no_grad()
    def predict_all(self, win):
        """win [H, FEAT] -> [M, 8] probabilities of every level."""
        return torch.cat([self.probs(m, win[None]) for m in range(self.M)])

    def record(self, win, raw, a_opp):
        self.buffer.append((win, raw, int(a_opp)))

    def buffer_tensors(self):
        """-> (wins [N, H, FEAT], raws long [N, 15], opponent actions long [N])."""
        wins, raws, acts = zip(*self.buffer)
        return torch.stack(wins), torch.stack(raws).long(), torch.tensor(acts)

    def finetune(self, m, wins, targets, steps=None):
        """`steps` (default grad_steps) Adam steps of cross-entropy to targets: [N] action ids or [N, 8] soft
        distributions."""
        model, opt = self.levels[m], self.opts[m]
        model.train()
        for _ in range(self.grad_steps if steps is None else int(steps)):
            logp = F.log_softmax(model(wins), dim=-1)
            loss = (F.nll_loss(logp, targets) if targets.dim() == 1 else -(targets * logp).sum(1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        return loss.item()

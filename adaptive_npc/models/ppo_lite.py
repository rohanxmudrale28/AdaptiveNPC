"""Minimal PPO for ONLINE updates of learning players during a match.

Trains an SB3 ActorCriticPolicy in place (the same object ppo_npc.load_policy returns), so pretrained
checkpoints need no conversion. Usage: record() one transition per step, update(last_value) when the
buffer should be consumed (last_value = V(next input), 0 if the round just ended).

cfg keys: lr, epochs, minibatch, clip, ent_coef, vf_coef, target_kl, max_grad_norm, gamma, gae_lambda,
critic_warmup_updates (first N updates train the value head only).
"""
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def action_probs(policy, x):
    """x: [B, D] float array/tensor -> action probabilities [B, N_ACTIONS] (torch)."""
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    return policy.get_distribution(x).distribution.probs


@torch.no_grad()
def probs_and_value(policy, x):
    """Single input x: [D] -> (probs np [N_ACTIONS], value float)."""
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))[None]
    probs = policy.get_distribution(t).distribution.probs[0].numpy().astype(np.float64)
    return probs / probs.sum(), float(policy.predict_values(t)[0, 0])


def entropy(probs):
    """Mean entropy (nats) of a [B, N] probability tensor."""
    return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1).mean())


def kl(p, q):
    """Mean KL(p || q) (nats) over rows of two [B, N] probability tensors."""
    p, q = p.clamp_min(1e-12), q.clamp_min(1e-12)
    return float((p * (torch.log(p) - torch.log(q))).sum(-1).mean())


class PPOLite:
    def __init__(self, policy, cfg, seed=0):
        self.policy, self.cfg = policy, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=float(cfg["lr"]))
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0
        self.clear()

    def clear(self):
        self.obs, self.act, self.logp, self.val, self.rew, self.done = [], [], [], [], [], []

    def __len__(self):
        return len(self.rew)

    def record(self, obs, action, logp, value, reward, done):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.act.append(int(action))
        self.logp.append(float(logp))
        self.val.append(float(value))
        self.rew.append(float(reward))
        self.done.append(bool(done))

    def _gae(self, last_value):
        c = self.cfg
        gamma, lam = float(c["gamma"]), float(c["gae_lambda"])
        n = len(self.rew)
        adv = np.zeros(n, dtype=np.float32)
        next_v, last = float(last_value), 0.0
        for i in reversed(range(n)):
            nonterm = 0.0 if self.done[i] else 1.0
            delta = self.rew[i] + gamma * next_v * nonterm - self.val[i]
            last = delta + gamma * lam * nonterm * last
            adv[i] = last
            next_v = self.val[i]
        return adv, adv + np.asarray(self.val, dtype=np.float32)

    def update(self, last_value=0.0):
        """Consume the buffer with clipped PPO. Returns a dict of update stats."""
        if len(self) == 0:
            return {}
        c = self.cfg
        adv, ret = self._gae(last_value)
        obs = torch.as_tensor(np.stack(self.obs))
        act = torch.as_tensor(self.act)
        old_logp = torch.as_tensor(self.logp, dtype=torch.float32)
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret)
        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        warmup = self.n_updates < int(c.get("critic_warmup_updates", 0))
        n, mb = len(act), int(c["minibatch"])
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "epochs_run": 0}
        self.policy.set_training_mode(True)
        stop = False
        for _ in range(int(c["epochs"])):
            perm = self.rng.permutation(n)
            for s in range(0, n, mb):
                idx = torch.as_tensor(perm[s:s + mb])
                values, logp, ent = self.policy.evaluate_actions(obs[idx], act[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv_t[idx]
                pg_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * a).mean()
                v_loss = ((values.flatten() - ret_t[idx]) ** 2).mean()
                ent_m = ent.mean()
                loss = float(c["vf_coef"]) * v_loss
                if not warmup:
                    loss = loss + pg_loss - float(c["ent_coef"]) * ent_m
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), float(c["max_grad_norm"]))
                self.opt.step()
                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - torch.log(ratio)).mean())
                stats.update(pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropy=ent_m.item(), approx_kl=approx_kl,
                             clip_frac=float((torch.abs(ratio - 1) > c["clip"]).float().mean()))
                if not warmup and approx_kl > float(c["target_kl"]):
                    stop = True
                    break
            stats["epochs_run"] += 1
            if stop:
                break
        self.policy.set_training_mode(False)
        self.n_updates += 1
        stats["warmup"] = warmup
        self.clear()
        return stats

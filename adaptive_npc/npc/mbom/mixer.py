"""Bayesian mixing of IOP levels (MBOM eq. 6-8).

After each real opponent action aᵒ:
    posterior_m ∝ (π̃_m(aᵒ|s) + eps) · prior_m          (eq. 6)
    prior       ← prior_avg · prior + (1 - prior_avg) · posterior, floored at prior_floor (moving average of
                  past posteriors; the floor keeps a level that lost for a long time recoverable after a switch)
    Ψ           ← decay · Ψ + (1 - decay) · posterior   (eq. 7, decayed average of posteriors; a distribution)
    α           = softmax(Ψ / T)                         (eq. 8, "softer" softmax)
Because Ψ sums to 1, α_max ≤ e^{1/T} / (e^{1/T} + M - 1) (≈ 0.79 for T = 0.5, M = 3): α never fully commits.
Modes: 'bayes' (above), 'uniform' (α ≡ 1/M), 'fixed' (α one-hot at fixed_level). State persists across rounds
within a match; reset() is for match start.
"""
import numpy as np

MODES = ("bayes", "uniform", "fixed")


class BayesMixer:
    def __init__(self, n_levels, decay=0.9, temperature=0.5, prior_avg=0.9, eps=1e-3, prior_floor=0.01,
                 mode="bayes", fixed_level=None):
        if mode not in MODES:
            raise ValueError(f"mixer mode must be one of {MODES}, got {mode!r}")
        if mode == "fixed" and not (fixed_level is not None and 0 <= int(fixed_level) < n_levels):
            raise ValueError(f"mode 'fixed' needs 0 <= fixed_level < {n_levels}, got {fixed_level!r}")
        self.M, self.mode = int(n_levels), mode
        self.fixed_level = None if fixed_level is None else int(fixed_level)
        self.decay, self.T, self.prior_avg = float(decay), float(temperature), float(prior_avg)
        self.eps, self.prior_floor = float(eps), float(prior_floor)
        self.reset()

    @property
    def alpha_cap(self):
        """Largest α any level can reach in 'bayes' mode (Ψ one-hot)."""
        e = np.exp(1.0 / self.T)
        return float(e / (e + self.M - 1))

    def reset(self):
        self.prior = np.full(self.M, 1.0 / self.M)
        self.psi = np.full(self.M, 1.0 / self.M)
        self.posterior = self.prior.copy()

    @property
    def alpha(self):
        if self.mode == "uniform":
            return np.full(self.M, 1.0 / self.M)
        if self.mode == "fixed":
            a = np.zeros(self.M)
            a[self.fixed_level] = 1.0
            return a
        z = self.psi / self.T
        e = np.exp(z - z.max())
        return e / e.sum()

    def update(self, lik):
        """lik: [M] probability each level gave to the observed opponent action. Returns the posterior."""
        lik = np.asarray(lik, dtype=np.float64)
        post = (lik + self.eps) * self.prior
        post /= post.sum()
        self.posterior = post
        prior = self.prior_avg * self.prior + (1 - self.prior_avg) * post
        prior = np.maximum(prior, self.prior_floor)
        self.prior = prior / prior.sum()
        self.psi = self.decay * self.psi + (1 - self.decay) * post
        return post

    def mix(self, level_probs):
        """level_probs: [M, n_actions] -> α-weighted mixture [n_actions]."""
        return self.alpha @ np.asarray(level_probs, dtype=np.float64)

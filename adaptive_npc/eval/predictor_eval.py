"""Predictor evaluation: per-archetype accuracy / KL vs baselines, entropy ceilings, and the accuracy
recovery curve after a style switch. Every model sees the identical stream (a tap around the NPC)."""
import warnings

import numpy as np

from adaptive_npc.env.actions import Action, N_ACTIONS
from adaptive_npc.env.match import run_match
from adaptive_npc.models.player_model import (MajorityBaseline, OnlinePredictor, RepeatLastBaseline)
from adaptive_npc.models.predictor_data import make_npc
from adaptive_npc.players.base import Player
from adaptive_npc.players.scripted import make_player

EPS = 1e-8
MODELS = ["gru_frozen", "gru_online", "majority", "repeat_last"]


def make_predictors(model, cfg):
    return {"gru_frozen": OnlinePredictor(model, cfg["model"]["window"]),
            "gru_online": OnlinePredictor(model, cfg["model"]["window"], cfg["online"]),
            "majority": MajorityBaseline(),
            "repeat_last": RepeatLastBaseline(cfg["eval"]["repeat_last_eps"])}


class PredictorTap(Player):
    """Wraps the NPC: predicts the player's action from the NPC-view obs before each step, then feeds
    the true action back. Records per-step probs for every predictor."""
    learns = True

    def __init__(self, npc, predictors):
        self.npc, self.predictors = npc, predictors
        self.probs = {k: [] for k in predictors}
        self.rnd, self.a_pl, self.rounds = [], [], -1

    def seed(self, seed):
        self.npc.seed(seed)

    def reset(self):
        self.rounds += 1
        self.npc.reset()
        for p in self.predictors.values():
            p.reset_round()

    def act(self, obs, hist):
        for k, p in self.predictors.items():
            self.probs[k].append(p.predict(obs))
        return self.npc.act(obs, hist)

    def observe(self, tr):
        self.rnd.append(self.rounds)
        self.a_pl.append(tr.opp_action)
        for p in self.predictors.values():
            p.observe(tr.action, tr.opp_action)


def run_tapped(env_cfg, players_cfg, cfg, model, npc_name, player, n_rounds, seed):
    tap = PredictorTap(make_npc(npc_name, players_cfg, env_cfg, seed), make_predictors(model, cfg))
    results = run_match(env_cfg, tap, player, n_rounds=n_rounds, seed=seed)
    return {"probs": {k: np.array(v) for k, v in tap.probs.items()}, "y": np.array(tap.a_pl),
            "round": np.array(tap.rnd), "results": results}


def acc_nll(probs, y):
    hit = probs.argmax(-1) == y
    nll = -np.log(probs[np.arange(len(y)), y] + EPS)
    return hit, nll


def mixer_dist(players_cfg):
    w = np.array([players_cfg["mixer"]["weights"][a.name.lower()] for a in Action], dtype=np.float64)
    return w / w.sum()


def archetype_ceiling(name, players_cfg):
    """(max achievable top-1 accuracy, entropy in nats or nan, note). Exact for the mixer; for the
    others only an upper bound that accounts for the epsilon-random 'noise' action."""
    if name == "mixer" and players_cfg["mixer"].get("noise", 0.0) == 0.0:
        p = mixer_dist(players_cfg)
        return float(p.max()), float(-(p * np.log(p)).sum()), "exact (i.i.d. mixer)"
    noise = float(players_cfg.get(name, {}).get("noise", 0.0))
    return 1.0 - noise + noise / N_ACTIONS, float("nan"), "upper bound (noise only)"


def per_archetype_rows(env_cfg, players_cfg, cfg, model, players=None):
    e = cfg["eval"]
    names = players or e["players"] or (list(players_cfg["train"]) + list(players_cfg["heldout"]))
    rows = []
    for npc_name in e["npcs"]:
        for pi, name in enumerate(names):
            hits = {k: [] for k in MODELS}
            nlls = {k: [] for k in MODELS}
            kl_true = {k: [] for k in MODELS}
            for seed in e["seeds"]:
                s = int(seed) * 131 + pi
                player = make_player(name, players_cfg, env_cfg, s + 1)
                out = run_tapped(env_cfg, players_cfg, cfg, model, npc_name, player, e["match_rounds"], s)
                for k in MODELS:
                    h, n = acc_nll(out["probs"][k], out["y"])
                    hits[k].append(h), nlls[k].append(n)
                    if name == "mixer" and players_cfg["mixer"].get("noise", 0.0) == 0.0:
                        p = mixer_dist(players_cfg)
                        kl_true[k].append((p * (np.log(p) - np.log(out["probs"][k] + EPS))).sum(-1))
            ceil, ent, note = archetype_ceiling(name, players_cfg)
            for k in MODELS:
                rows.append({"npc": npc_name, "archetype": name, "heldout": name in players_cfg["heldout"],
                             "model": k, "top1_acc": float(np.concatenate(hits[k]).mean()),
                             "kl_nll": float(np.concatenate(nlls[k]).mean()),
                             "kl_to_true": float(np.concatenate(kl_true[k]).mean()) if kl_true[k] else float("nan"),
                             "ceiling_acc": ceil, "entropy_nats": ent, "ceiling_note": note,
                             "steps": int(sum(len(h) for h in hits[k]))})
    return rows


def recovery_rows(env_cfg, players_cfg, cfg, model):
    """Accuracy vs steps since each style switch (bin_steps wide), per target archetype and pooled.
    bin_start = -1 is the pre-switch reference (last `pre_rounds` rounds before the switch)."""
    sw = cfg["eval"]["switch"]
    sp = players_cfg["switcher"]
    every, cycle = int(sp["switch_every_rounds"]), list(sp["cycle"])
    bw, horizon, pre = int(sw["bin_steps"]), int(sw["horizon_steps"]), int(sw["pre_rounds"])
    n_bins = horizon // bw
    acc = {}  # (target, model) -> list of per-event arrays [n_bins + 1] (index 0 = pre-switch)
    for seed in sw["seeds"]:
        player = make_player("switcher", players_cfg, env_cfg, int(seed) + 1)
        out = run_tapped(env_cfg, players_cfg, cfg, model, sw["npc"], player, sw["match_rounds"], int(seed))
        rnd, y = out["round"], out["y"]
        hits = {k: acc_nll(out["probs"][k], y)[0] for k in MODELS}
        for r0 in range(every, sw["match_rounds"], every):
            target = cycle[(r0 // every) % len(cycle)]
            seg = rnd >= r0
            seg_end = rnd < r0 + every
            idx = np.where(seg & seg_end)[0]
            pre_idx = np.where((rnd >= r0 - pre) & (rnd < r0))[0]
            if len(idx) == 0 or len(pre_idx) == 0:
                continue
            bins = (np.arange(len(idx)) // bw)
            for k in MODELS:
                row = np.full(n_bins + 1, np.nan)
                row[0] = hits[k][pre_idx].mean()
                for b in range(n_bins):
                    m = bins == b
                    if m.any():
                        row[b + 1] = hits[k][idx[m]].mean()
                for tgt in (target, "all"):
                    acc.setdefault((tgt, k), []).append(row)
    rows = []
    for (tgt, k), evs in sorted(acc.items()):
        arr = np.array(evs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # bins with no data in any event
            mean = np.nanmean(arr, axis=0)
        for i in range(n_bins + 1):
            rows.append({"target": tgt, "bin_start": -1 if i == 0 else (i - 1) * bw, "model": k,
                         "acc": float(mean[i]), "n_events": int((~np.isnan(arr[:, i])).sum())})
    return rows

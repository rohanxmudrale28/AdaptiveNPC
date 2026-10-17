"""Γ error: one-step and 2-step (open-loop, Γ's own snapped state fed back) per state channel and for reward.

Data is a dict of arrays in time order: state, next_state (N, 15), a_npc, a_opp, reward, done, ep (episode id),
npc_id, opp_id (+ npc_names / opp_names). A 2-step pair is row i -> i + 1 inside one episode with row i non-terminal;
the true actions of both rows are replayed through the model and compared with row i + 1's next_state / reward.
Rows: mae / rmse in raw units (facing: mae is the class mismatch rate, rmse empty), exact = fraction exactly right
(reward: within `reward_tol`), copy_exact = exact rate of the 'nothing changes' baseline (state kept, t not compared).
"""
import numpy as np

from adaptive_npc.env.actions import N_ACTIONS
from adaptive_npc.models.env_model import CHANNELS, CHANNEL_IDX, FACE, TrueSimulator


def _channel_rows(pred, true, base, reward_pred, reward_true, tol, done_true):
    n = len(true)
    rows = []
    for name, c in zip(CHANNELS, CHANNEL_IDX):
        d = pred[:, c].astype(float) - true[:, c]
        face = c in FACE
        rows.append({"feature": name, "n": n, "mae": float(np.mean(d != 0)) if face else float(np.abs(d).mean()),
                     "rmse": float("nan") if face else float(np.sqrt((d ** 2).mean())),
                     "exact": float(np.mean(d == 0)), "copy_exact": float(np.mean(base[:, c] == true[:, c]))})
    idx = list(CHANNEL_IDX)
    rows.append({"feature": "all_state", "n": n, "mae": float("nan"), "rmse": float("nan"),
                 "exact": float(np.mean((pred[:, idx] == true[:, idx]).all(1))),
                 "copy_exact": float(np.mean((base[:, idx] == true[:, idx]).all(1)))})
    pos = [c for c in CHANNEL_IDX[:4]]
    rows.append({"feature": "position_all4", "n": n, "mae": float("nan"), "rmse": float("nan"),
                 "exact": float(np.mean((pred[:, pos] == true[:, pos]).all(1))),
                 "copy_exact": float(np.mean((base[:, pos] == true[:, pos]).all(1)))})
    dr = reward_pred.astype(float) - reward_true
    rows.append({"feature": "reward", "n": n, "mae": float(np.abs(dr).mean()), "rmse": float(np.sqrt((dr ** 2).mean())),
                 "exact": float(np.mean(np.abs(dr) <= tol)), "copy_exact": float(np.mean(np.abs(reward_true) <= tol))})
    if done_true.any():
        rows.append({"feature": "reward_terminal", "n": int(done_true.sum()), "mae": float(np.abs(dr[done_true]).mean()),
                     "rmse": float(np.sqrt((dr[done_true] ** 2).mean())),
                     "exact": float(np.mean(np.abs(dr[done_true]) <= tol)),
                     "copy_exact": float(np.mean(np.abs(reward_true[done_true]) <= tol))})
    return rows


def predictions(model, data):
    """One-step and 2-step predictions: dict of aligned arrays per horizon."""
    s, n = data["state"], data["next_state"]
    s1, r1, pd1 = model.step(s, data["a_npc"], data["a_opp"])
    out = {1: {"idx": np.arange(len(s)), "pred": s1, "pred_r": r1, "pred_done": pd1, "true": n, "true_r": data["reward"],
               "base": s, "done": data["done"].astype(bool)}}
    i = np.where((data["ep"][:-1] == data["ep"][1:]) & ~data["done"][:-1].astype(bool))[0]
    if len(i):
        s2, r2, _ = model.step(s1[i], data["a_npc"][i + 1], data["a_opp"][i + 1])
        out[2] = {"idx": i + 1, "pred": s2, "pred_r": r2, "true": n[i + 1], "true_r": data["reward"][i + 1],
                  "base": s[i], "done": data["done"][i + 1].astype(bool)}
    return out


def random_action_predictions(model, data, env_cfg, n, seed):
    """Random-action slice: n states sampled from `data` (e.g. val), BOTH sides' actions uniformly random at each of
    the two steps. Ground truth comes from the true simulator; 2-step pairs whose first step ends the round are dropped."""
    rng = np.random.default_rng(seed)
    s = data["state"][rng.choice(len(data["state"]), min(n, len(data["state"])), replace=False)]
    a = rng.integers(N_ACTIONS, size=(4, len(s)))
    sim = TrueSimulator(env_cfg)
    t1, tr1, td1 = sim.step(s, a[0], a[1])
    p1, pr1, pd1 = model.step(s, a[0], a[1])
    k = ~td1
    t2, tr2, td2 = sim.step(t1[k], a[2][k], a[3][k])
    p2, pr2, _ = model.step(p1[k], a[2][k], a[3][k])
    return {1: {"pred": p1, "pred_r": pr1, "pred_done": pd1, "true": t1, "true_r": tr1, "base": s, "done": td1},
            2: {"pred": p2, "pred_r": pr2, "true": t2, "true_r": tr2, "base": s[k], "done": td2}}


def rows_from_predictions(preds, reward_tol):
    rows = []
    for h, p in preds.items():
        rows += [{"horizon": h, **r} for r in _channel_rows(p["pred"], p["true"], p["base"], p["pred_r"], p["true_r"],
                                                            reward_tol, p["done"])]
    return rows


def evaluate(model, data, reward_tol):
    """Rows of {horizon, feature, n, mae, rmse, exact, copy_exact} for horizons 1 and 2."""
    return rows_from_predictions(predictions(model, data), reward_tol)


def evaluate_random_actions(model, data, env_cfg, n, seed, reward_tol):
    """(feature rows, errors_by_group row) for the random-action slice."""
    preds = random_action_predictions(model, data, env_cfg, n, seed)
    return rows_from_predictions(preds, reward_tol), _group_row("random_actions", "val_states", preds, reward_tol)


def _group_row(group, name, preds, tol, mask=None):
    idx_all = list(CHANNEL_IDX)
    row = {"group": group, "name": name}
    for h, p in preds.items():
        m = np.ones(len(p["pred"]), bool) if mask is None else mask(h, p)
        if not m.any():
            return None
        dr = np.abs(p["pred_r"][m] - p["true_r"][m])
        row.update({f"n_h{h}": int(m.sum()),
                    f"all_state_exact_h{h}": float(np.mean((p["pred"][m][:, idx_all] == p["true"][m][:, idx_all]).all(1))),
                    f"reward_mae_h{h}": float(dr.mean()), f"reward_exact_h{h}": float(np.mean(dr <= tol))})
    return row


def evaluate_groups(model, data, reward_tol):
    """Per NPC kind / per opponent: all-state exact rate and reward MAE / exactness at both horizons."""
    preds = predictions(model, data)
    rows = []
    for kind, ids, names in (("npc", data["npc_id"], data["npc_names"]), ("opponent", data["opp_id"], data["opp_names"])):
        for g, name in enumerate(names):
            row = _group_row(kind, str(name), {h: p for h, p in preds.items() if (ids[p["idx"]] == g).any()}, reward_tol,
                             lambda h, p, ids=ids, g=g: ids[p["idx"]] == g)
            if row and "n_h1" in row:
                rows.append(row)
    return rows


def format_table(rows, title):
    """Markdown table: one row per feature; columns = 1-step / 2-step MAE and exact rate + copy baseline."""
    by = {(r["horizon"], r["feature"]): r for r in rows}
    feats = [r["feature"] for r in rows if r["horizon"] == 1]
    n1 = by[(1, feats[0])]["n"]
    n2 = by[(2, feats[0])]["n"] if (2, feats[0]) in by else 0
    lines = [f"**{title}** (1-step n={n1}, 2-step n={n2}; face_* 'MAE' = mismatch rate; reward 'exact' = within tol)", "",
             "| feature | 1-step MAE | 1-step exact | 2-step MAE | 2-step exact | copy-state exact (1 / 2) |",
             "|---|---|---|---|---|---|"]
    for f in feats:
        a, b = by[(1, f)], by.get((2, f))
        mae = lambda r: "" if r is None or np.isnan(r["mae"]) else f"{r['mae']:.4f}"  # noqa: E731
        ex = lambda r: "" if r is None else f"{100 * r['exact']:.2f}%"  # noqa: E731
        cp = f"{100 * a['copy_exact']:.1f}% / " + ("" if b is None else f"{100 * b['copy_exact']:.1f}%")
        lines.append(f"| {f} | {mae(a)} | {ex(a)} | {mae(b)} | {ex(b)} | {cp} |")
    return "\n".join(lines)


TERMINAL_CATEGORIES = ("ko_win", "ko_loss", "ko_draw", "timeout_win", "timeout_loss", "timeout_draw",
                       "win", "loss", "draw", "timeout", "all_round_end")


def terminal_breakdown(preds, env_cfg, reward_tol):
    """One-step reward error on round-end steps, by true outcome (NPC view). ko_* = a player reached 0 hp, timeout_* =
    the step limit ended it; win / loss / draw / timeout are the totals. Rows: category, n, reward_mae, reward_exact
    (within tol), done_correct (Γ's predicted round end matches), outcome_correct (the sign of the win / loss / draw
    bonus implied by the predicted state matches)."""
    p, max_steps = preds[1], int(env_cfg["max_steps"])
    t, d = p["true"], p["done"]
    ko = (t[:, 6] <= 0) | (t[:, 7] <= 0)
    res = np.where(t[:, 6] > t[:, 7], 1, np.where(t[:, 7] > t[:, 6], -1, 0))
    pd_ = p["pred"]
    pres = np.where(pd_[:, 6] > pd_[:, 7], 1, np.where(pd_[:, 7] > pd_[:, 6], -1, 0))
    dr = np.abs(p["pred_r"] - p["true_r"])
    masks = {f"{k}_{o}": d & (ko if k == "ko" else ~ko) & (res == v) for k in ("ko", "timeout")
             for o, v in (("win", 1), ("loss", -1), ("draw", 0))}
    masks.update({"win": d & (res == 1), "loss": d & (res == -1), "draw": d & (res == 0), "timeout": d & ~ko,
                  "all_round_end": d})
    rows = []
    for c in TERMINAL_CATEGORIES:
        m = masks[c]
        rows.append({"category": c, "n": int(m.sum()),
                     "reward_mae": float(dr[m].mean()) if m.any() else float("nan"),
                     "reward_exact": float(np.mean(dr[m] <= reward_tol)) if m.any() else float("nan"),
                     "done_correct": float(np.mean(p["pred_done"][m])) if m.any() else float("nan"),
                     "outcome_correct": float(np.mean(pres[m] == res[m])) if m.any() else float("nan")})
    return rows


def format_terminal_table(rows_by_set):
    lines = ["**Round-end reward error by outcome (1-step)**", "",
             "| set | outcome | n | reward MAE | reward exact | round end predicted | outcome predicted |",
             "|---|---|---|---|---|---|---|"]
    for name, rows in rows_by_set.items():
        for r in rows:
            f = lambda v, pct=True: "" if np.isnan(v) else (f"{100 * v:.1f}%" if pct else f"{v:.4f}")  # noqa: E731
            lines.append(f"| {name} | {r['category']} | {r['n']} | {f(r['reward_mae'], False)} | {f(r['reward_exact'])} | "
                         f"{f(r['done_correct'])} | {f(r['outcome_correct'])} |")
    return chr(10).join(lines)

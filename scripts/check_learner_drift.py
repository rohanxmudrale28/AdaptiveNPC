"""M9: do the learning players change during a match vs a frozen NPC, and do they beat, match or lose to it?

python -m scripts.check_learner_drift --config configs/learners.yaml [--entry lr1e-3_u64] [--set match.rounds=100]
python -m scripts.check_learner_drift --config configs/learners.yaml --reaggregate   # rebuild outcome/chosen/plots only

For every frozen NPC opponent x learner (naive, reasoning) x seed:
  1. probe set: the frozen start learner plays warm-up rounds (own seed) -> `probe.states` states
  2. `match.matches_per_seed` independent frozen-control matches and, per sweep entry, learning matches. Each
     starts from the pretrained weights on its own env seed. After every round: KL(pi_start || pi_now) on the
     probe set, policy entropy, NPC-prediction accuracy
     -> <out>/<opp>/<learner>[/<entry>]/seed_<n>[/match_<j>]/rounds.csv  (match_<j> only if matches_per_seed > 1)
  3. collapse guards: the frozen start / end learner plays `guards.eval_rounds` rounds vs the NPC, the end
     learner also vs a random NPC.
Writes <out>/drift.csv, match_summary.csv, outcome.csv (beat/match/lose per opponent, CI over all matches),
chosen.csv (LR choice: KL + random guard on kl_check_opponent, collapse guard on collapse_check_opponent), plots/*.png.
"""
import argparse
import csv
import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from adaptive_npc.config import add_config_args, config_from_args, load_config  # noqa: E402
from adaptive_npc.env.gym_env import NPCEnv  # noqa: E402
from adaptive_npc.eval.metrics import mean_ci  # noqa: E402
from adaptive_npc.eval.runner import build_npc, play_match, write_csv  # noqa: E402
from adaptive_npc.logging_utils import save_run_metadata  # noqa: E402
from adaptive_npc.models.ppo_lite import entropy, kl  # noqa: E402
from adaptive_npc.players.naive_learner import NaiveLearner  # noqa: E402
from adaptive_npc.players.reasoning_learner import ReasoningLearner  # noqa: E402
from adaptive_npc.seeding import seed_everything  # noqa: E402

LEARNERS = ("naive", "reasoning")


def learner_score(winner):
    """Learner sits in slot 1: win 1, draw 1/2, loss 0."""
    return 1.0 if winner == 1 else 0.5 if winner == -1 else 0.0


def make_learner(cfg, kind, seed, entry, learn):
    lc = cfg["learners"]
    ppo = {**cfg["ppo_lite"], "lr": float(entry["lr"])}
    ue = int(entry["update_every"])
    if kind == "naive":
        return NaiveLearner(lc["naive"]["policy"].format(seed=seed), ppo, ue, learn, seed)
    return ReasoningLearner(lc["reasoning"]["policy"].format(seed=seed), lc["reasoning"]["npc_model"],
                            ppo, ue, lc["gru_online"], learn, seed)


class Arena:
    """Builds matches of a learner (slot 1) vs a frozen NPC (slot 0) through NPCEnv + runner.build_npc."""

    def __init__(self, cfg, env_cfg, players_cfg):
        self.cfg, self.env_cfg, self.players_cfg = cfg, env_cfg, players_cfg
        n = cfg["npc"]
        self.npc_cfg = {"eval": {"deterministic": bool(n["deterministic"]), "predictor_online": False},
                        "online": None, "predictor_checkpoint": n["predictor_checkpoint"],
                        "train": {"model_dir": n["model_dir"]}}

    def rounds(self, method, learner, n_rounds, env_seed, ckpt_seed, after_round=None):
        """Play n_rounds; after_round(i, row) is called after each round. Returns run_match-style rows."""
        base = NPCEnv(self.env_cfg, opponent=learner, seed=env_seed)
        env, act_fn, on_rs = build_npc(self.npc_cfg, self.env_cfg, self.players_cfg, method, ckpt_seed, base,
                                       env_seed + 2)
        out = []
        for i in range(n_rounds):
            row = play_match(env, act_fn, 1, env_seed if i == 0 else None, on_rs)[0]
            row["round"] = i
            if after_round:
                after_round(i, row)
            out.append(row)
        return out

    def score(self, method, learner, n_rounds, env_seed, ckpt_seed):
        return float(np.mean([learner_score(r["winner"]) for r in self.rounds(method, learner, n_rounds, env_seed, ckpt_seed)]))


def collect_probe(arena, method, start, cfg, env_seed, ckpt_seed):
    """Frozen start learner plays warm-up rounds until it has seen probe.states states; subsample them."""
    p = cfg["probe"]
    lrn = start.frozen_copy()
    lrn.record_inputs = []
    base = NPCEnv(arena.env_cfg, opponent=lrn, seed=env_seed)
    env, act_fn, on_rs = build_npc(arena.npc_cfg, arena.env_cfg, arena.players_cfg, method, ckpt_seed, base, env_seed + 2)
    for i in range(int(p["max_rounds"])):
        play_match(env, act_fn, 1, env_seed if i == 0 else None, on_rs)
        if len(lrn.record_inputs) >= int(p["states"]):
            break
    recs = lrn.record_inputs
    idx = np.sort(np.random.default_rng(env_seed).choice(len(recs), min(int(p["states"]), len(recs)), replace=False))
    obs = np.stack([recs[i][0] for i in idx])
    win = torch.stack([recs[i][1] for i in idx]) if recs[0][1] is not None else None
    return {"obs": obs, "win": win}


@torch.no_grad()
def npc_model_probs(lrn, probe):
    return F.softmax(lrn.predictor.model(probe["win"]), dim=-1)


def probe_inputs(probe, gru_owner):
    """Policy inputs on the probe set; the reasoning learner's prediction comes from gru_owner's GRU."""
    if probe["win"] is None:
        return probe["obs"]
    return np.concatenate([probe["obs"], npc_model_probs(gru_owner, probe).numpy()], axis=1)


def drift_stats(start, cur, probe):
    p_start = start.policy_probs(probe_inputs(probe, start))
    p_full = cur.policy_probs(probe_inputs(probe, cur))
    out = {"kl_full": kl(p_start, p_full), "kl_policy": kl(p_start, cur.policy_probs(probe_inputs(probe, start))),
           "entropy": entropy(p_full), "npc_model_kl": float("nan")}
    if probe["win"] is not None:
        out["npc_model_kl"] = kl(npc_model_probs(start, probe), npc_model_probs(cur, probe))
    return out


def round_row(r, hp_max):
    return {"round": r["round"], "winner": r["winner"], "score": learner_score(r["winner"]),
            "hp_learner": r["hp_player"], "hp_npc": r["hp_npc"], "damage_dealt": hp_max - r["hp_npc"],
            "damage_taken": hp_max - r["hp_player"], "steps": r["steps"], "learner_return": -r["ret_npc"]}


def play_tracked(arena, method, lrn, start, probe, n_rounds, env_seed, ckpt_seed, hp_max):
    """Match with per-round drift tracking. Returns per-round rows."""
    rows = []

    def after(i, r):
        row = round_row(r, hp_max)
        row.update(drift_stats(start, lrn, probe))
        row["n_updates"] = lrn.n_updates
        hits = getattr(lrn, "pred_hits", None)
        if hits is not None:
            new = hits[after.n_hits:]
            after.n_hits = len(hits)
            row["npc_pred_acc"] = float(np.mean(new)) if new else float("nan")
        else:
            row["npc_pred_acc"] = float("nan")
        rows.append(row)
    after.n_hits = 0
    arena.rounds(method, lrn, n_rounds, env_seed, ckpt_seed, after)
    return rows


def summarize_match(rows, first_n):
    s = np.array([r["score"] for r in rows])
    n = min(first_n, len(s))
    return {"score": float(s.mean()), "score_first_n": float(s[:n].mean()), "score_last_n": float(s[-n:].mean()),
            "score_delta": float(s[-n:].mean() - s[:n].mean()),
            "learner_return": float(np.mean([r["learner_return"] for r in rows])),
            "damage_dealt": float(np.mean([r["damage_dealt"] for r in rows])),
            "damage_taken": float(np.mean([r["damage_taken"] for r in rows])),
            "npc_pred_acc": float(np.nanmean([r["npc_pred_acc"] for r in rows]))
            if any(not np.isnan(r["npc_pred_acc"]) for r in rows) else float("nan")}


def verdict(mean, ci):
    if np.isnan(ci):
        return ("beat" if mean > 0.5 else "lose" if mean < 0.5 else "match") + " (1 seed)"
    if mean - ci > 0.5:
        return "beat"
    if mean + ci < 0.5:
        return "lose"
    return "match"


def aggregate_outcome(match_rows):
    groups = {}
    for r in match_rows:
        groups.setdefault((r["opponent"], r["learner"], r["entry"]), []).append(r)
    out = []
    for (opp, lrn, entry), rs in groups.items():
        row = {"opponent": opp, "learner": lrn, "entry": entry, "n_seeds": len({r["seed"] for r in rs}),
               "n_matches": len(rs)}  # CIs are over all matches (seeds x matches_per_seed)
        for m in ("score", "score_first_n", "score_last_n", "score_delta", "learner_return", "damage_dealt",
                  "damage_taken", "npc_pred_acc"):
            row[f"{m}_mean"], row[f"{m}_ci95"] = mean_ci([r[m] for r in rs])
        row["verdict"] = verdict(row["score_mean"], row["score_ci95"])
        out.append(row)
    return out


def checks(rs, kl_min, tol):
    """Mean-over-matches checks on drift rows of one (opponent, learner, entry)."""
    kl_m = float(np.mean([r["kl_full"] for r in rs]))
    rand = float(np.mean([r["end_vs_random"] for r in rs]))
    start, end = float(np.mean([r["start_vs_npc"] for r in rs])), float(np.mean([r["end_vs_npc"] for r in rs]))
    return {"kl": kl_m, "vs_random": rand, "start_vs_npc": start, "end_vs_npc": end,
            "kl_pass": kl_m >= kl_min, "random_pass": rand > 0.5, "collapse_pass": end >= start - tol}


def choose(drift_rows, kl_opp, collapse_opp, entries, kl_min, tol):
    """Per learner: smallest LR (then larger update_every) such that, averaged over all seeds x matches,
    KL >= kl_min and end-vs-random > 0.5 on kl_opp, and end-vs-NPC >= start - tol on collapse_opp.
    The collapse check uses a beatable opponent: vs ppo_predicted the start score is ~0 and it cannot fail."""
    out = []
    order = sorted(entries, key=lambda e: (float(e["lr"]), -int(e["update_every"])))
    opps = sorted({r["opponent"] for r in drift_rows})
    for kind in LEARNERS:
        row = {"learner": kind, "chosen_entry": "none", "kl_check_opponent": kl_opp, "collapse_check_opponent": collapse_opp}
        for e in order:
            sel = {o: [r for r in drift_rows if r["learner"] == kind and r["entry"] == e["name"] and r["opponent"] == o]
                   for o in opps}
            if not sel.get(kl_opp) or not sel.get(collapse_opp):
                continue
            k, c = checks(sel[kl_opp], kl_min, tol), checks(sel[collapse_opp], kl_min, tol)
            if k["kl_pass"] and k["random_pass"] and c["collapse_pass"]:
                per_opp = {o: checks(rs, kl_min, tol) for o, rs in sel.items() if rs}
                row.update(chosen_entry=e["name"], lr=float(e["lr"]), update_every=int(e["update_every"]),
                           kl=k["kl"], end_vs_random=k["vs_random"],
                           collapse_start=c["start_vs_npc"], collapse_end=c["end_vs_npc"],
                           passes_all_opponents=all(v["kl_pass"] and v["random_pass"] and v["collapse_pass"]
                                                    for v in per_opp.values()))
                break
        out.append(row)
    keys = list(dict.fromkeys(k for r in out for k in r))
    return [{k: r.get(k, "") for k in keys} for r in out]


# fixed categorical order for sweep entries (validated palette, light mode); frozen control in secondary ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def plot_opponent(path, opp, curves, entries, kl_min, chosen, smooth):
    """curves[(learner, entry)] = list of per-match per-round row lists. Small multiples: learner x metric."""
    metrics = [("score", "learner score (win 1, draw ½)"), ("kl_full", "KL(π_start ‖ π_now), nats"),
               ("entropy", "policy entropy, nats")]
    fig, axes = plt.subplots(len(LEARNERS), len(metrics), figsize=(13, 6.5), squeeze=False, facecolor=SURFACE)
    n_runs = 0
    for i, kind in enumerate(LEARNERS):
        for j, (m, label) in enumerate(metrics):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            series = [(f"frozen_{kind}", "frozen control", MUTED, "--", None)]
            series += [(e["name"], e["name"], SERIES[k % len(SERIES)], "-", MARKERS[k % len(MARKERS)])
                       for k, e in enumerate(entries)]
            focus = {f"frozen_{kind}", chosen.get(kind)}  # these get full weight + CI band; other settings recede
            for key, name, color, ls, mk in series:
                runs = curves.get((kind, key))
                if not runs:
                    continue
                n = min(len(r) for r in runs)
                vals = np.array([[row[m] for row in r[:n]] for r in runs])
                if m == "score" and smooth > 1:  # trailing rolling mean per match; CSVs stay raw
                    kern = np.ones(smooth)
                    vals = np.array([np.convolve(v, kern)[:n] / np.minimum(np.arange(1, n + 1), smooth) for v in vals])
                y = vals.mean(axis=0)
                x = np.arange(1, n + 1)
                strong = key in focus
                ax.plot(x, y, color=color, ls=ls, lw=2.2 if strong else 1.2, alpha=1.0 if strong else 0.55,
                        marker=mk, ms=4 if strong else 2.5, label=name, zorder=3 if strong else 2)
                if m == "score" and strong and len(runs) > 1:  # 95% t-interval over matches, per round
                    half = np.array([mean_ci(vals[:, t])[1] for t in range(n)])
                    ax.fill_between(x, np.clip(y - half, 0, 1), np.clip(y + half, 0, 1), color=color, alpha=0.12, lw=0)
                n_runs = max(n_runs, len(runs))
            if m == "kl_full":
                ax.axhline(kl_min, color=MUTED, lw=1, ls=":", zorder=1)
                ax.annotate(f"kl_min {kl_min:g}", (0.99, kl_min), xycoords=("axes fraction", "data"), ha="right",
                            xytext=(0, 3), textcoords="offset points", fontsize=7, color=MUTED)
            if m == "score":
                label = f"score, {smooth}-round mean" if smooth > 1 else label
                if chosen.get(kind):
                    label += f" (chosen {chosen[kind]})"
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, color="#e6e5e0", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_title(f"{kind}: {label}", fontsize=9, color=INK, loc="left")
            if i == len(LEARNERS) - 1:
                ax.set_xlabel("round", fontsize=8, color=MUTED)
            if m == "score":
                ax.set_ylim(-0.05, 1.05)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(f"Learners vs frozen {opp}: mean over {n_runs} matches per line. Score: win 1, draw ½; "
                 f"bold = frozen control + chosen setting, band = 95% CI over matches",
                 fontsize=11, color=INK, y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=min(len(labels), 7),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--entry", action="append", default=None, help="run only these sweep entries")
    parser.add_argument("--reaggregate", action="store_true",
                        help="no matches: rebuild outcome.csv, chosen.csv and plots from saved match_summary/drift CSVs")
    args = parser.parse_args()
    cfg = config_from_args(args)
    torch.set_num_threads(1)
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    hp_max = int(env_cfg["hp"])
    out = Path(args.out or cfg["out_dir"])
    entries = cfg.get("sweep") or [{"name": "main", "lr": cfg["ppo_lite"]["lr"], "update_every": cfg["update_every"]}]
    if args.entry:
        entries = [e for e in entries if e["name"] in args.entry]
    m, n_cfg = cfg["match"], cfg["npc"]
    n_rounds, first_n = int(m["rounds"]), int(m["first_n"])
    n_matches = int(m.get("matches_per_seed", 1))
    if args.reaggregate:
        return finish(cfg, out, entries, read_csv(out / "match_summary.csv"), read_csv(out / "drift.csv"), time.time())
    arena = Arena(cfg, env_cfg, players_cfg)
    save_run_metadata(out, cfg)
    drift_rows, match_rows = [], []
    t0 = time.time()
    for opp in n_cfg["opponents"]:
        for kind in LEARNERS:
            for seed in m["seeds"]:
                seed = int(seed)
                ckpt_seed = seed if n_cfg["checkpoint_seed"] is None else int(n_cfg["checkpoint_seed"])
                env_seed = int(m["env_seed_base"]) + seed
                g_seed = env_seed + int(cfg["guards"]["seed_offset"])
                n_eval = int(cfg["guards"]["eval_rounds"])
                seed_everything(env_seed)
                start = make_learner(cfg, kind, seed, entries[0], learn=False)
                probe = collect_probe(arena, opp, start, cfg, env_seed + int(cfg["probe"]["seed_offset"]), ckpt_seed)
                start_vs_npc = arena.score(opp, start.frozen_copy(), n_eval, g_seed, ckpt_seed)
                h_start = drift_stats(start, start, probe)["entropy"]

                runs = [(f"frozen_{kind}", f"frozen_{kind}", entries[0], False)]
                runs += [(kind, e["name"], e, True) for e in entries]
                for (lname, ename, e, learn), j in itertools.product(runs, range(n_matches)):
                    # each match restarts from the pretrained weights on its own env seed; probe + guard seeds are shared
                    match_seed = env_seed + int(m.get("match_seed_stride", 1000)) * j
                    seed_everything(match_seed)
                    lrn = make_learner(cfg, kind, seed, e, learn)
                    rows = play_tracked(arena, opp, lrn, start, probe, n_rounds, match_seed, ckpt_seed, hp_max)
                    cell = out / opp / lname / ("" if not learn else ename) / f"seed_{seed}"
                    if n_matches > 1:
                        cell = cell / f"match_{j}"
                    save_run_metadata(cell, {**cfg, "opponent": opp, "learner": lname, "entry": e, "seed": seed, "match": j})
                    write_csv(cell / "rounds.csv", rows)
                    match_rows.append({"opponent": opp, "learner": lname, "entry": ename if learn else "-", "seed": seed,
                                       "match": j, **summarize_match(rows, first_n)})
                    if not learn:
                        continue
                    end = lrn.frozen_copy()
                    end_vs_npc = arena.score(opp, end, n_eval, g_seed, ckpt_seed)
                    end_vs_random = arena.score("random", lrn.frozen_copy(), n_eval, g_seed + 1, ckpt_seed)
                    last = rows[-1]
                    drift_rows.append({
                        "opponent": opp, "learner": kind, "entry": ename, "lr": float(e["lr"]),
                        "update_every": int(e["update_every"]), "seed": seed, "match": j, "kl_full": last["kl_full"],
                        "kl_policy": last["kl_policy"], "npc_model_kl": last["npc_model_kl"],
                        "entropy_start": h_start,
                        "entropy_end": last["entropy"], "n_updates": lrn.n_updates,
                        "kl_pass": last["kl_full"] >= float(cfg["kl_min"]),
                        "start_vs_npc": start_vs_npc, "end_vs_npc": end_vs_npc, "end_vs_random": end_vs_random,
                        "guard_random_pass": end_vs_random > 0.5,
                        "guard_collapse_pass": end_vs_npc >= start_vs_npc - float(cfg["guards"]["collapse_tol"])})
                    print(f"[{opp:>13} {kind:>9} {ename:>12} s{seed} m{j}] score {match_rows[-1]['score']:.2f} "
                          f"kl {last['kl_full']:.3f} H {last['entropy']:.2f} end/start vs npc {end_vs_npc:.2f}/{start_vs_npc:.2f} "
                          f"vs random {end_vs_random:.2f} ({time.time() - t0:.0f}s)", flush=True)

    write_csv(out / "drift.csv", drift_rows)
    write_csv(out / "match_summary.csv", match_rows)
    finish(cfg, out, entries, match_rows, drift_rows, t0)


def finish(cfg, out, entries, match_rows, drift_rows, t0):
    """outcome.csv, chosen.csv and plots from per-match rows (also used by --reaggregate on saved CSVs)."""
    outcome = aggregate_outcome(match_rows)
    write_csv(out / "outcome.csv", outcome)
    chosen = choose(drift_rows, cfg["kl_check_opponent"], cfg.get("collapse_check_opponent", cfg["kl_check_opponent"]),
                    entries, float(cfg["kl_min"]), float(cfg["guards"]["collapse_tol"]))
    write_csv(out / "chosen.csv", chosen)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    for opp in cfg["npc"]["opponents"]:
        plot_opponent(out / "plots" / f"{opp}.png", opp, curves_for(out, opp, entries, cfg["match"]["seeds"]), entries,
                      float(cfg["kl_min"]), {c["learner"]: c["chosen_entry"] for c in chosen if c["chosen_entry"] != "none"},
                      int(cfg.get("plot_smooth_rounds", 5)))

    print("\noutcome (learner score vs frozen NPC; >0.5 = learner ahead; CI over all matches):")
    for r in outcome:
        print(f"  {r['opponent']:>13} {r['learner']:>16} {r['entry']:>12}  score {r['score_mean']:.2f}±{r['score_ci95']:.2f}  "
              f"last-first {r['score_delta_mean']:+.2f}  n={r['n_matches']}  -> {r['verdict']}")
    print("chosen:", ", ".join(f"{c['learner']}={c['chosen_entry']}" for c in chosen))
    print(f"wrote {out}/outcome.csv, chosen.csv, plots/ ({time.time() - t0:.0f}s)")


def read_csv(path):
    """Saved per-match CSV -> rows with numbers / booleans parsed back."""
    def parse(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except ValueError:
            return v
    with open(path, encoding="utf-8") as f:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def curves_for(out, opp, entries, seeds):
    """Reload this opponent's rounds.csv files -> {(learner, entry): [rows per match]}.
    Layout: seed_<n>/rounds.csv (matches_per_seed 1) or seed_<n>/match_<j>/rounds.csv."""
    curves = {}
    for kind in LEARNERS:
        for key, sub in [(f"frozen_{kind}", Path(f"frozen_{kind}"))] + [(e["name"], Path(kind) / e["name"]) for e in entries]:
            runs = []
            for s in seeds:
                d = out / opp / sub / f"seed_{int(s)}"
                files = [d / "rounds.csv"] if (d / "rounds.csv").exists() else sorted(
                    d.glob("match_*/rounds.csv"), key=lambda p: int(p.parent.name.split("_")[1]))
                for f in files:
                    with open(f, encoding="utf-8") as fh:
                        runs.append([{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)])
            curves[(kind, key)] = runs
    return curves


if __name__ == "__main__":
    main()

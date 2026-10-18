"""M10 final grid (criteria pre-registered in PLAN.md, M10). Final claims: exploiter_seed0 / exploiter_seed2 only.

python -m scripts.final_grid --config configs/mbom.yaml [--methods mbom] [--opponents exploiter_seed0] [--seeds 0 1]
                             [--threads 1] [--aggregate-only]
Every (method, opponent, seed) part = matches_per_cell matches of `rounds` rounds, fresh NPC / opponent state per
match, the same env / opponent / agent seeds per match for every method (paired). Parts are resumable and can run in
parallel processes; failures are logged (error.txt) and reported. Aggregate -> <final.out_dir>/:
  methods.csv      per method x opponent: win rate (draws = 1/2), rounds 1-5 / last 5, damage taken, MBOM trace means
  comparisons.csv  per primary comparison x group: paired mean +- 95% CI, paired t-test p, Holm-adjusted p within the
                   group's primary family, verdict; secondary paired damage-taken difference
Groups: every opponent (all seeds); every exploiter at the seed it was trained against (seed 0) and at seeds 1-4;
the exploiters pooled at seed 0 and at seeds 1-4.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.eval.metrics import mean_ci
from adaptive_npc.eval.runner import write_csv
from adaptive_npc.seeding import seed_everything
from scripts.dev_variants import load_rows, run_parts, summarize


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired(rows_a, rows_b, key):
    b = {(r["opponent"], r["seed"], r["match"]): float(r[key]) for r in rows_b}
    return np.array([float(r[key]) - b[k] for r in rows_a if (k := (r["opponent"], r["seed"], r["match"])) in b])


def groups(fc):
    tr = str(fc["exploiter_train_seed"])
    out = [(opp, lambda r, o=opp: r["opponent"] == o) for opp in fc["opponents"]]
    for e in fc["exploiters"]:
        out += [(f"{e}@seed{tr}", lambda r, e=e: r["opponent"] == e and r["seed"] == tr),
                (f"{e}@other_seeds", lambda r, e=e: r["opponent"] == e and r["seed"] != tr)]
    ex = set(fc["exploiters"])
    out += [(f"exploiters@seed{tr}", lambda r: r["opponent"] in ex and r["seed"] == tr),
            ("exploiters@other_seeds", lambda r: r["opponent"] in ex and r["seed"] != tr)]
    return out


def aggregate(cfg):
    fc = cfg["final"]
    root = Path(fc["out_dir"])
    rows, status = {}, []
    for m in fc["methods"]:
        rows[m], missing, failed, sec = load_rows(root, m, fc["opponents"], fc["seeds"])
        status.append({"method": m, "status": "failed" if failed else ("incomplete" if missing else "ok"),
                       "cpu_minutes": round(sec / 60, 1), "errors": " | ".join(failed + missing)[:500]})
    write_csv(root / "status.csv", status)
    methods = [{"method": m, "opponent": o, **summarize([r for r in rows[m] if r["opponent"] == o])}
               for m in fc["methods"] for o in fc["opponents"]]
    write_csv(root / "methods.csv", methods)
    comps = []
    for gname, sel in groups(fc):
        fam = []
        for a, b in fc["primary"]:
            ra, rb = [r for r in rows[a] if sel(r)], [r for r in rows[b] if sel(r)]
            d, dd = paired(ra, rb, "win_rate"), paired(ra, rb, "damage_taken")
            mean, ci = mean_ci(d) if len(d) else (float("nan"), float("nan"))
            p = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 and np.std(d) > 0 else (1.0 if len(d) else float("nan"))
            dm, dci = mean_ci(dd) if len(dd) else (float("nan"), float("nan"))
            fam.append({"group": gname, "comparison": f"{a} - {b}", "n_matches": len(d), "paired_mean": mean,
                        "paired_ci95": ci, "p": p, "damage_diff": dm, "damage_diff_ci95": dci})
        valid = [i for i, f in enumerate(fam) if not np.isnan(f["p"])]
        adj = holm([fam[i]["p"] for i in valid]) if valid else []
        for i, f in enumerate(fam):
            f["p_holm"] = float(adj[valid.index(i)]) if i in valid else float("nan")
            sig = f["p_holm"] < 0.05
            f["verdict"] = ("no data" if np.isnan(f["p"]) else
                            "better" if sig and f["paired_mean"] > 0 else
                            "worse" if sig and f["paired_mean"] < 0 else "no detectable difference")
            f["practically_significant"] = bool(abs(f["paired_mean"]) >= float(fc["min_gain"])) if f["n_matches"] else ""
        comps += fam
    write_csv(root / "comparisons.csv", comps)
    print(f"{'method':22s}{'status':>12s}{'cpu':>10s}")
    for s in status:
        print(f"{s['method']:22s}{s['status']:>12s}{s['cpu_minutes']:>10.1f} cpu-min  {s['errors'][:80]}")
    print(f"\n{'group':28s}{'comparison':40s}{'n':>5s}{'paired':>9s}{'+-95%':>7s}{'p_holm':>8s}  verdict")
    for c in comps:
        if c["group"] in fc["opponents"] and c["group"] not in fc["exploiters"]:
            continue  # per-opponent rows are in the CSV; print the exploiter claim groups
        print(f"{c['group']:28s}{c['comparison']:40s}{c['n_matches']:>5d}{c['paired_mean']:>+9.3f}{c['paired_ci95']:>7.3f}"
              f"{c['p_holm']:>8.3f}  {c['verdict']}")
    print(f"wrote {root}")


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--opponents", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    fc = cfg["final"]
    if "exploiter_seed1" in fc["opponents"]:
        raise ValueError("exploiter_seed1 is the development exploiter and is excluded from the final grid")
    if not args.aggregate_only:
        for m in args.methods or fc["methods"]:
            run_parts(cfg, env_cfg, players_cfg, fc["out_dir"], m, [], fc,
                      args.seeds if args.seeds is not None else fc["seeds"], args.opponents or fc["opponents"])
    aggregate(cfg)


if __name__ == "__main__":
    main()

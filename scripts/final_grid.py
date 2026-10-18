"""M10 final grid (criteria pre-registered in PLAN.md, M10). Final claims: exploiter_seed0 / exploiter_seed2 only.

python -m scripts.final_grid --config configs/mbom.yaml [--h2-passed] [--reduced-seeds] [--threads 1]
                             [--methods mbom] [--opponents exploiter_seed0] [--seeds 0 1] [--aggregate-only]
Amended design (PLAN.md M10, time limit): archetypes at single_matches_per_cell (1), exploiters and learners at
matches_per_cell (7); methods = `methods`, plus `methods_if_h2` with --h2-passed (refused unless the H2 summary reports
SUPPORTED); --reduced-seeds puts the exploiter claim groups on reduced_claim_seeds (0-2). Pass the same flags to
--aggregate-only. Every (method, opponent, seed) part = its matches of `rounds` rounds, fresh NPC / opponent state per
match, the same env / opponent / agent seeds per match for every method (paired). Parts are resumable and can run in
parallel processes; failures are logged (error.txt) and reported. Aggregate -> <final.out_dir>/:
  methods.csv      per method x opponent: win rate (draws = 1/2), rounds 1-5 / last 5, damage taken, MBOM trace means
  comparisons.csv  per primary comparison x group: paired mean +- 95% CI, paired t-test p, Holm-adjusted p within the
                   group's primary family, verdict; secondary paired damage-taken difference
Groups: every opponent (all seeds); every exploiter at the seed it was trained against (seed 0) and at seeds 1-4;
the exploiters pooled at seed 0 and at seeds 1-4.
"""
import argparse
import csv
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


def h2_supported(fc):
    p = Path(fc.get("h2_summary") or "")
    if not fc.get("h2_summary") or not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return next(csv.DictReader(f)).get("hypothesis_supported") == "True"


def resolve_plan(fc, h2_passed, reduced_seeds):
    """Amended design (PLAN.md M10): methods / comparisons (conditional on H2) and per-opponent matches and seeds."""
    supported = h2_supported(fc)
    if h2_passed and supported is not True:
        raise ValueError(f"--h2-passed, but {fc['h2_summary']} does not report H2 as supported ({supported})")
    if not h2_passed and supported:
        print("note: H2 is reported as supported; the pre-registration says to run with --h2-passed")
    single = set(fc.get("single_match_opponents", []))
    claim_seeds = fc["reduced_claim_seeds"] if reduced_seeds else fc["seeds"]
    n_single = int(fc.get("single_matches_per_cell", fc["matches_per_cell"]))
    return {"methods": list(fc["methods"]) + (list(fc.get("methods_if_h2", [])) if h2_passed else []),
            "primary": list(fc["primary"]) + (list(fc.get("primary_if_h2", [])) if h2_passed else []),
            "secondary": list(fc.get("secondary_if_h2", [])) if h2_passed else [],
            "matches": {o: n_single if o in single else int(fc["matches_per_cell"])
                        for o in fc["opponents"]},
            "seeds": {o: list(claim_seeds) if o in fc["exploiters"] else list(fc["seeds"]) for o in fc["opponents"]},
            "h2_passed": bool(h2_passed), "reduced_seeds": bool(reduced_seeds)}


def compare(rows, a, b, sel):
    ra, rb = [r for r in rows.get(a, []) if sel(r)], [r for r in rows.get(b, []) if sel(r)]
    d, dd = paired(ra, rb, "win_rate"), paired(ra, rb, "damage_taken")
    mean, ci = mean_ci(d) if len(d) else (float("nan"), float("nan"))
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 and np.std(d) > 0 else (1.0 if len(d) else float("nan"))
    dm, dci = mean_ci(dd) if len(dd) else (float("nan"), float("nan"))
    return {"comparison": f"{a} - {b}", "n_matches": len(d), "paired_mean": mean, "paired_ci95": ci, "p": p,
            "damage_diff": dm, "damage_diff_ci95": dci}


def aggregate(cfg, plan):
    fc = cfg["final"]
    root = Path(fc["out_dir"])
    rows, status = {}, []
    for m in plan["methods"]:
        rows[m], missing, failed, sec = [], [], [], 0.0
        for o in fc["opponents"]:
            r, mi, fa, s = load_rows(root, m, [o], plan["seeds"][o])
            rows[m] += r
            missing += mi
            failed += fa
            sec += s
        status.append({"method": m, "status": "failed" if failed else ("incomplete" if missing else "ok"),
                       "cpu_minutes": round(sec / 60, 1), "errors": " | ".join(failed + missing)[:500],
                       "h2_passed": plan["h2_passed"], "reduced_seeds": plan["reduced_seeds"]})
    write_csv(root / "status.csv", status)
    methods = [{"method": m, "opponent": o, "matches_per_cell": plan["matches"][o],
                "seeds": " ".join(map(str, plan["seeds"][o])), **summarize([r for r in rows[m] if r["opponent"] == o])}
               for m in plan["methods"] for o in fc["opponents"]]
    write_csv(root / "methods.csv", methods)
    comps = []
    for gname, sel in groups(fc):
        fam = [{"group": gname, "family": "primary", **compare(rows, a, b, sel)} for a, b in plan["primary"]]
        valid = [i for i, f in enumerate(fam) if not np.isnan(f["p"])]
        adj = holm([fam[i]["p"] for i in valid]) if valid else []
        for i, f in enumerate(fam):
            f["p_holm"] = float(adj[valid.index(i)]) if i in valid else float("nan")
            sig = f["p_holm"] < 0.05
            f["verdict"] = ("no data" if np.isnan(f["p"]) else
                            "better" if sig and f["paired_mean"] > 0 else
                            "worse" if sig and f["paired_mean"] < 0 else "no detectable difference")
            f["practically_significant"] = bool(abs(f["paired_mean"]) >= float(fc["min_gain"])) if f["n_matches"] else ""
        for a, b in plan["secondary"]:  # outside the Holm family: unadjusted p, no verdict
            s = {"group": gname, "family": "secondary", **compare(rows, a, b, sel)}
            s.update({"p_holm": float("nan"), "verdict": "secondary (unadjusted)", "practically_significant":
                      bool(abs(s["paired_mean"]) >= float(fc["min_gain"])) if s["n_matches"] else ""})
            fam.append(s)
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
    parser.add_argument("--h2-passed", action="store_true",
                        help="add methods_if_h2 / primary_if_h2 / secondary_if_h2 (checked against h2_summary)")
    parser.add_argument("--reduced-seeds", action="store_true", help="exploiter claim groups on reduced_claim_seeds")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    fc = cfg["final"]
    if "exploiter_seed1" in fc["opponents"]:
        raise ValueError("exploiter_seed1 is the development exploiter and is excluded from the final grid")
    plan = resolve_plan(fc, args.h2_passed, args.reduced_seeds)
    if not args.aggregate_only:
        # expensive methods first, so identical parallel processes (part locks) finish together
        order = sorted(args.methods or plan["methods"], key=lambda m: not m.startswith("mbom_") or m == "mbom_no_imagination")
        for m in order:
            for opp in args.opponents or fc["opponents"]:
                design = {**fc, "matches_per_cell": plan["matches"][opp]}
                seeds = [s for s in plan["seeds"][opp] if args.seeds is None or s in args.seeds]
                run_parts(cfg, env_cfg, players_cfg, fc["out_dir"], m, [], design, seeds, [opp])
    aggregate(cfg, plan)


if __name__ == "__main__":
    main()

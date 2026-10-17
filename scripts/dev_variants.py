"""MBOM development variants (DEVELOPMENT SET ONLY: exploiter_seed1 + train archetypes) and the adaptive-opponent test.

python -m scripts.dev_variants --config configs/mbom.yaml [--variants top_level uniform_mix] [--designs exploiter]
                               [--seeds 0 1] [--threads 1] [--aggregate-only]
python -m scripts.dev_variants --config configs/mbom.yaml --adaptive [--methods mbom] [--opponents switcher] ...

Each variant = runner method + config overrides (dev_variants.variants), played with the gonogo match design (fresh NPC
state per match, same env / opponent / agent seeds per match for every variant, so variants are paired match by
match). Parts are resumable (a finished part is skipped) and can run in parallel processes. Variants with `reuse`
take the saved go/no-go runs for the exploiter design instead of re-running them (same seeds, deterministic).
A failing part is logged (error.txt) and the run continues. The aggregate writes dev_variants.csv: one row per
variant x design with status, win rates, MBOM trace means and paired differences vs the reference variants.
--adaptive runs adaptive_test (MBOM vs no_imagination vs ppo_predicted vs learners / switcher) -> summary.csv.
"""
import argparse
import csv
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from adaptive_npc.config import add_config_args, apply_overrides, config_from_args, load_config
from adaptive_npc.eval.metrics import mean_ci
from adaptive_npc.eval.runner import write_csv
from adaptive_npc.seeding import seed_everything
from scripts.eval_mbom import prepare
from scripts.gonogo_mbom import read_csv, run_part

METRICS = ("win_rate", "win_rate_early", "win_rate_late", "damage_taken", "mean_steps", "acc_mixed", "argmax_changed",
           "input_entropy", "phi0_entropy")


def design_opponents(design, players_cfg):
    return design["opponents"] or list(players_cfg["train"])


def part_dir(root, opp, method, seed):
    return Path(root) / opp / method / f"seed_{seed}"


def run_parts(cfg, env_cfg, players_cfg, root, method, overrides, design, seeds, opponents):
    """Play every (opponent, seed) part not yet on disk. Returns the list of errors."""
    c = prepare(apply_overrides(cfg, overrides))
    errors = []
    for opp in opponents:
        for seed in seeds:
            cell = part_dir(root, opp, method, seed)
            if (cell / "matches.csv").exists():
                continue
            c["gonogo"] = {**cfg["gonogo"], "opponent": opp, "matches_per_cell": design["matches_per_cell"],
                           "rounds": design["rounds"], "env_seed_base": design["env_seed_base"]}
            t0 = time.perf_counter()
            try:
                run_part(c, env_cfg, players_cfg, method, seed, Path(root) / opp)
                (cell / "seconds.txt").write_text(f"{time.perf_counter() - t0:.1f}\n")
            except Exception:  # log and continue: every tried variant is reported, including failures
                cell.mkdir(parents=True, exist_ok=True)
                (cell / "error.txt").write_text(traceback.format_exc())
                errors.append(f"{opp}/seed_{seed}")
                print(f"  FAILED {method} {opp} seed {seed} (see {cell / 'error.txt'})")
    return errors


def load_rows(root, method, opponents, seeds):
    rows, missing, failed, seconds = [], [], [], 0.0
    for opp in opponents:
        for seed in seeds:
            cell = part_dir(root, opp, method, seed)
            if (cell / "matches.csv").exists():
                rows += [{**r, "opponent": opp} for r in read_csv(cell / "matches.csv")]
                if (cell / "seconds.txt").exists():
                    seconds += float((cell / "seconds.txt").read_text())
            elif (cell / "error.txt").exists():
                failed.append(f"{opp}/seed_{seed}: {(cell / 'error.txt').read_text().strip().splitlines()[-1]}")
            else:
                missing.append(f"{opp}/seed_{seed}")
    return rows, missing, failed, seconds


def paired_diff(rows_a, rows_b, key="win_rate"):
    b = {(r["opponent"], r["seed"], r["match"]): float(r[key]) for r in rows_b}
    d = [float(r[key]) - b[k] for r in rows_a if (k := (r["opponent"], r["seed"], r["match"])) in b]
    return (*mean_ci(d), len(d)) if d else (float("nan"), float("nan"), 0)


def summarize(rows):
    out = {"n_matches": len(rows)}
    for k in METRICS:
        vals = [float(r[k]) for r in rows if r.get(k, "") not in ("", None)]
        out[k] = float(np.mean(vals)) if vals else ""
    return out


def variant_root(dv, v, design_name):
    return Path(dv["out_dir"]) / v["name"] / design_name


def reuse_root(v, design_name):
    return (v.get("reuse") or {}).get(design_name)


def aggregate_variants(cfg, players_cfg):
    dv = cfg["dev_variants"]
    log, loaded = [], {}
    for dname, design in dv["designs"].items():
        opps = design_opponents(design, players_cfg)
        for v in dv["variants"]:
            root = reuse_root(v, dname) or variant_root(dv, v, dname)
            if reuse_root(v, dname):  # go/no-go layout: <root>/<method>/seed_<n>
                rows, missing, failed, sec = [], [], [], 0.0
                for seed in design["seeds"]:
                    p = Path(root) / v["method"] / f"seed_{seed}" / "matches.csv"
                    if p.exists():
                        rows += [{**r, "opponent": opps[0]} for r in read_csv(p)]
                    else:
                        missing.append(str(p))
            else:
                rows, missing, failed, sec = load_rows(root, v["method"], opps, design["seeds"])
            loaded[(v["name"], dname)] = rows
            status = "failed" if failed else ("incomplete" if missing else "ok")
            log.append({"variant": v["name"], "method": v["method"], "overrides": " ".join(v["overrides"]),
                        "debug_not_reported": bool(v.get("debug", False)), "design": dname,
                        "opponents": " ".join(opps), "status": status,
                        "errors": " | ".join(failed + [f"missing {m}" for m in missing])[:500],
                        "source": "reused go/no-go run" if reuse_root(v, dname) else str(root),
                        "cpu_minutes": round(sec / 60, 1), **summarize(rows)})
    refs = [dv["reference"]] + list(dv["compare"])
    for row in log:
        for ref in refs:
            mean, ci, n = paired_diff(loaded[(row["variant"], row["design"])], loaded.get((ref, row["design"]), []))
            row[f"diff_vs_{ref}"], row[f"diff_vs_{ref}_ci95"], row[f"n_paired_vs_{ref}"] = mean, ci, n
    write_csv(dv["log"], log)
    print(f"\n{'variant':20s}{'design':11s}{'status':11s}{'n':>4s}{'win':>7s}{'early':>7s}{'late':>7s}{'dmg':>6s}"
          f"{'acc':>6s}{'chg':>6s}{'vs mbom':>14s}{'vs no_imag':>14s}{'vs ppo_pred':>14s}")
    for r in log:
        f = lambda x, w=7: f"{x:>{w}.3f}" if isinstance(x, float) else f"{'':>{w}s}"  # noqa: E731
        diffs = "".join(f"{r[f'diff_vs_{ref}']:>+8.3f}±{r[f'diff_vs_{ref}_ci95']:<5.3f}" if r[f"n_paired_vs_{ref}"]
                        else f"{'':>14s}" for ref in refs)
        tag = " (debug)" if r["debug_not_reported"] else ""
        print(f"{r['variant'] + tag:20s}{r['design']:11s}{r['status']:11s}{r['n_matches']:>4d}{f(r['win_rate'])}"
              f"{f(r['win_rate_early'])}{f(r['win_rate_late'])}{f(r['damage_taken'], 6)}{f(r['acc_mixed'], 6)}"
              f"{f(r['argmax_changed'], 6)}{diffs}")
    print(f"wrote {dv['log']}")


def aggregate_adaptive(cfg):
    at = cfg["adaptive_test"]
    root = Path(at["out_dir"])
    rows = {m: load_rows(root, m, at["opponents"], at["seeds"])[0] for m in at["methods"]}
    out = []
    for opp in at["opponents"] + ["pooled"]:
        sel = {m: [r for r in rs if opp == "pooled" or r["opponent"] == opp] for m, rs in rows.items()}
        row = {"opponent": opp}
        for m in at["methods"]:
            s = summarize(sel[m])
            row.update({f"{m}_win_rate": s["win_rate"], f"{m}_early": s["win_rate_early"],
                        f"{m}_late": s["win_rate_late"], f"{m}_damage": s["damage_taken"], f"{m}_n": s["n_matches"]})
        for ref in ("mbom_no_imagination", "ppo_predicted"):
            mean, ci, n = paired_diff(sel["mbom"], sel[ref])
            dmg, dmg_ci, _ = paired_diff(sel["mbom"], sel[ref], "damage_taken")  # secondary: win rate may saturate
            row.update({f"mbom_minus_{ref}": mean, f"mbom_minus_{ref}_ci95": ci, f"n_paired_{ref}": n,
                        f"mbom_minus_{ref}_damage": dmg, f"mbom_minus_{ref}_damage_ci95": dmg_ci})
        out.append(row)
    write_csv(root / "summary.csv", out)
    print(f"\n{'opponent':18s}" + "".join(f"{m:>22s}" for m in at["methods"])
          + f"{'mbom - no_imag':>18s}{'mbom - ppo_pred':>18s}")
    for r in out:
        print(f"{r['opponent']:18s}" + "".join(
            f"{r[f'{m}_win_rate']:>12.3f} ({r[f'{m}_early']:.2f}>{r[f'{m}_late']:.2f})" if r[f"{m}_n"] else f"{'':>22s}"
            for m in at["methods"])
            + f"{r['mbom_minus_mbom_no_imagination']:>+11.3f}±{r['mbom_minus_mbom_no_imagination_ci95']:<6.3f}"
            + f"{r['mbom_minus_ppo_predicted']:>+11.3f}±{r['mbom_minus_ppo_predicted_ci95']:<6.3f}"
            + f"   damage taken mbom {r['mbom_damage']:.2f}, vs no_imag "
            f"{r['mbom_minus_mbom_no_imagination_damage']:+.2f}±{r['mbom_minus_mbom_no_imagination_damage_ci95']:.2f}")
    print(f"wrote {root / 'summary.csv'}")


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--designs", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--methods", nargs="+", default=None, help="--adaptive only")
    parser.add_argument("--opponents", nargs="+", default=None, help="--adaptive only")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    if args.adaptive:
        at = cfg["adaptive_test"]
        if not args.aggregate_only:
            for m in args.methods or at["methods"]:
                run_parts(cfg, env_cfg, players_cfg, at["out_dir"], m, [], at,
                          args.seeds if args.seeds is not None else at["seeds"], args.opponents or at["opponents"])
        aggregate_adaptive(cfg)
        return
    dv = cfg["dev_variants"]
    if not args.aggregate_only:
        for v in dv["variants"]:
            if args.variants and v["name"] not in args.variants:
                continue
            for dname, design in dv["designs"].items():
                if (args.designs and dname not in args.designs) or reuse_root(v, dname):
                    continue
                print(f"variant {v['name']} / {dname}")
                run_parts(cfg, env_cfg, players_cfg, variant_root(dv, v, dname), v["method"], v["overrides"], design,
                          args.seeds if args.seeds is not None else design["seeds"],
                          design_opponents(design, players_cfg))
    aggregate_variants(cfg, players_cfg)


if __name__ == "__main__":
    main()

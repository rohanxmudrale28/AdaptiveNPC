"""Diagnose a go/no-go run from its saved CSVs (no new matches).

python -m scripts.diagnose_mbom --config configs/mbom.yaml [--out results/mbom/gonogo]
Reads <gonogo.out_dir>/<method>/seed_<n>/{rounds.csv, mbom_rounds.csv} and writes <out>/diagnosis/:
  phases.csv         per method x phase (rounds 1-5 / 6-25 / 26-30): step-weighted per-level and mixed IOP accuracy,
                     α, policy input drift (entropy / max-prob vs φ0, argmax-change rate), update-weighted Q spread
                     and the implied max soft-BR target probability e^{s/τ} / (e^{s/τ} + 7)
  phases_by_seed.csv the same per NPC seed
  first_update.csv   per match: round and step of the first IOP update; whether MBOM and no_imagination played
                     identical rounds before it (they should: all levels equal φ0 until then)
  early.csv          rounds 1-5 win rate per method (and paired MBOM - no_imagination)
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from adaptive_npc.config import add_config_args, config_from_args
from adaptive_npc.eval.runner import write_csv

PHASES = (("1-5", 0, 5), ("6-25", 5, 25), ("26-30", 25, 30))
WEIGHTED = ("acc_l0", "acc_l1", "acc_l2", "acc_mixed", "nll_mixed", "alpha_l0_mean", "alpha_l1_mean",
            "alpha_l2_mean", "input_entropy", "phi0_entropy", "input_maxprob", "phi0_maxprob", "argmax_changed")


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def win_score(w):
    return 1.0 if w == 0 else 0.5 if w == -1 else 0.0


def phase_rows(trace, tau, keys):
    out = []
    for name, lo, hi in PHASES:
        rs = [r for r in trace if lo <= int(r["round"]) < hi]
        w = np.array([float(r["steps"]) for r in rs])
        row = {**keys, "phase": name, "rounds": len(rs), "steps": float(w.sum())}
        row.update({k: float(np.average([float(r[k]) for r in rs], weights=w)) for k in WEIGHTED})
        up = [(float(r["q_spread"]), int(r["updates"])) for r in rs if int(r["updates"]) > 0 and r["q_spread"] != "nan"]
        s = float(np.average([q for q, _ in up], weights=[n for _, n in up])) if up else float("nan")
        row["q_spread"] = s
        row["updates"] = int(sum(int(r["updates"]) for r in rs))
        row["br_target_maxprob_bound"] = float(np.exp(s / tau) / (np.exp(s / tau) + 7)) if up else float("nan")
        out.append(row)
    return out


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    g, tau = cfg["gonogo"], float(cfg["imagination"]["tau_br"])
    src = Path(args.out or g["out_dir"])
    out = src / "diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    trace_methods = [m for m in g["methods"] if m.startswith("mbom")]
    phases, by_seed, first, early = [], [], [], []
    for method in trace_methods:
        allt = []
        for seed in g["seeds"]:
            t = [{**r, "seed": seed} for r in read(src / method / f"seed_{seed}" / "mbom_rounds.csv")]
            allt += t
            by_seed += phase_rows(t, tau, {"method": method, "seed": seed})
        phases += phase_rows(allt, tau, {"method": method, "seed": "all"})
    write_csv(out / "phases.csv", phases)
    write_csv(out / "phases_by_seed.csv", by_seed)

    ue = int(cfg["mbom"]["update_every"])
    for seed in g["seeds"]:
        rounds = {m: read(src / m / f"seed_{seed}" / "rounds.csv") for m in g["methods"]}
        for i in range(int(g["matches_per_cell"])):
            rs = {m: [r for r in rounds[m] if int(r["match"]) == i] for m in g["methods"]}
            steps = np.cumsum([int(r["steps"]) for r in rs["mbom"]])
            fr = int(np.searchsorted(steps, ue))  # round in which real step `ue` falls
            same = all(rs["mbom"][k] == rs["mbom_no_imagination"][k] for k in range(fr))
            first.append({"seed": seed, "match": i, "first_update_round": fr + 1, "first_update_step": ue,
                          "rounds_before_identical_to_no_imagination": same,
                          "rounds_1_5_steps": int(steps[min(4, len(steps) - 1)])})
            for m in g["methods"]:
                early.append({"method": m, "seed": seed, "match": i,
                              "win_rate_rounds_1_5": float(np.mean([win_score(int(r["winner"])) for r in rs[m][:5]]))})
    write_csv(out / "first_update.csv", first)
    write_csv(out / "early.csv", early)

    print(f"{'method':22s}{'phase':>7s}{'acc_l0':>8s}{'acc_l1':>8s}{'acc_l2':>8s}{'acc_mix':>8s}{'a_l0':>6s}{'a_l1':>6s}"
          f"{'a_l2':>6s}{'H_in':>6s}{'H_phi0':>7s}{'max_in':>7s}{'max_phi0':>9s}{'chg':>6s}{'q_spr':>7s}{'br_max':>7s}")
    for r in phases:
        print(f"{r['method']:22s}{r['phase']:>7s}{r['acc_l0']:>8.3f}{r['acc_l1']:>8.3f}{r['acc_l2']:>8.3f}"
              f"{r['acc_mixed']:>8.3f}{r['alpha_l0_mean']:>6.2f}{r['alpha_l1_mean']:>6.2f}{r['alpha_l2_mean']:>6.2f}"
              f"{r['input_entropy']:>6.2f}{r['phi0_entropy']:>7.2f}{r['input_maxprob']:>7.2f}{r['phi0_maxprob']:>9.2f}"
              f"{r['argmax_changed']:>6.3f}{r['q_spread']:>7.3f}{r['br_target_maxprob_bound']:>7.3f}")
    fu = np.array([f["first_update_round"] for f in first])
    print(f"\nfirst update: round {np.median(fu):.0f} median (range {fu.min()}-{fu.max()}); identical rounds before it "
          f"in {sum(f['rounds_before_identical_to_no_imagination'] for f in first)}/{len(first)} matches")
    for m in g["methods"]:
        print(f"rounds 1-5 win rate {m:22s}{np.mean([e['win_rate_rounds_1_5'] for e in early if e['method'] == m]):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Train the env model Gamma and report its error on held-out transitions and held-out opponents.

python -m scripts.train_env_model --config configs/env_model.yaml [--recollect] [--eval-only]
python -m scripts.train_env_model --config configs/env_model.yaml --eval-only --set oracle_model=true   # debug: true simulator, must be 100%

Data: scripts.collect_env_data (cached in data/). Trains on `train`, picks the epoch with the best `val` loss,
then evaluates one-step and 2-step errors per state channel + reward on train / val (held-out transitions, same
opponents, different seeds) / heldout_opp (opponents never seen in training).
Writes <out_dir>/errors.csv, errors_by_group.csv, error_table.md, train_log.csv (+ config.yaml, git_hash.txt)
and the checkpoint <model_path> (gitignored).
"""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.eval.env_model_eval import (evaluate, evaluate_groups, evaluate_random_actions, format_table,
                                              format_terminal_table, predictions, random_action_predictions,
                                              terminal_breakdown)
from adaptive_npc.models.env_model_data import augment_random_actions, mix
from adaptive_npc.eval.runner import write_csv
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.models.env_model import EnvModel, EnvSpec, features, load_env_model, loss_fn, save_env_model, targets
from adaptive_npc.seeding import seed_everything
from scripts.collect_env_data import load_all


def tensors(spec, d):
    return features(spec, d["state"], d["a_npc"], d["a_opp"]), targets(spec, d["state"], d["next_state"], d["reward"])


@torch.no_grad()
def eval_loss(model, x, tg, cfg_t, chunk=65536):
    model.eval()
    cls, r = tg
    tot, n = 0.0, len(x)
    for i in range(0, n, chunk):
        out = model(x[i:i + chunk])
        l, _ = loss_fn(model.spec, out, cls[i:i + chunk], r[i:i + chunk], cfg_t["reward_loss_weight"])
        tot += float(l) * len(out)
    return tot / n


def fit(cfg, spec, data, log_path=None):
    t = cfg["train"]
    seed_everything(int(cfg["seed"]))
    model = EnvModel(spec, cfg["model"]["hidden"])
    train = data["train"]
    aug = cfg.get("aug")
    if aug and aug["enabled"]:
        assert int(aug["seed"]) != int(cfg["eval"]["random_actions"]["seed"]), "aug seed must differ from the eval slice seed"
        extra = augment_random_actions(spec.cfg, train["state"], len(train["state"]), float(aug["random_fraction"]),
                                       float(aug["two_step_share"]), float(aug["hard_weight"]), int(aug["seed"]))
        n0, n1 = len(train["state"]), len(extra["state"])
        print(f"train mix: {n0} on-policy + {n1} random-action ({100 * n1 / (n0 + n1):.1f}%)", flush=True)
        train = mix(train, extra)
        write_csv(Path(log_path).with_name("train_mix.csv"),
                  [{"source": "on_policy", "n": n0}, {"source": "random_action", "n": n1}]) if log_path else None
    x, tg = tensors(spec, train)
    xv, tgv = tensors(spec, data["val"])
    opt = torch.optim.Adam(model.parameters(), lr=float(t["lr"]))
    epochs, bs = int(t["epochs"]), int(t["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=float(t["lr_final"]))
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    best, best_state, log = float("inf"), None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=g)
        run = 0.0
        for i in range(0, len(x), bs):
            b = perm[i:i + bs]
            loss, _ = loss_fn(spec, model(x[b]), tg[0][b], tg[1][b], t["reward_loss_weight"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss) * len(b)
        sched.step()
        vl = eval_loss(model, xv, tgv, t)
        log.append({"epoch": ep, "train_loss": run / len(x), "val_loss": vl, "lr": sched.get_last_lr()[0]})
        print(f"epoch {ep:3d}  train {run / len(x):.5f}  val {vl:.5f}", flush=True)
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    if log_path:
        write_csv(log_path, log)
    return model.eval()


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--recollect", action="store_true", help="ignore cached data")
    parser.add_argument("--eval-only", action="store_true", help="skip training; evaluate the saved checkpoint")
    args = parser.parse_args()
    cfg = config_from_args(args)
    env_cfg = load_config(cfg["env_config"])
    spec = EnvSpec(env_cfg)
    out = Path(args.out or cfg["out_dir"])
    save_run_metadata(out, cfg)
    data = load_all(cfg, args.recollect)
    for s, d in data.items():
        print(f"{s}: {len(d['state'])} transitions, {int(d['ep'].max()) + 1} rounds, "
              f"opponents {list(d['opp_names'])}", flush=True)
    if cfg["oracle_model"]:
        print("oracle_model=true: evaluating the TRUE SIMULATOR (debug only)")
        model = load_env_model(None, oracle_model=True, env_cfg=env_cfg)
    elif args.eval_only:
        model = load_env_model(cfg["model_path"])
    else:
        model = fit(cfg, spec, data, out / "train_log.csv")
        Path(cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
        save_env_model(cfg["model_path"], model, env_cfg, {"seed": int(cfg["seed"])})
        print(f"saved {cfg['model_path']}")
    tol = float(cfg["eval"]["reward_tol"])
    rows, group_rows, tables = [], [], []
    titles = {"train": "train (seen transitions)", "val": "val (held-out transitions, same opponents)",
              "heldout_opp": "held-out opponents (never in train)"}
    for split in ("val", "heldout_opp", "train"):
        if split not in data:
            continue
        res = evaluate(model, data[split], tol)
        rows += [{"set": split, **r} for r in res]
        group_rows += [{"set": split, **r} for r in evaluate_groups(model, data[split], tol)]
        tables.append(format_table(res, titles.get(split, split)))
    ra = cfg["eval"]["random_actions"]
    res, grow = evaluate_random_actions(model, data["val"], env_cfg, int(ra["n"]), int(ra["seed"]), tol)
    rows += [{"set": "random_actions", **r} for r in res]
    group_rows.append({"set": "random_actions", **grow})
    tables.insert(2, format_table(res, f"random-action slice ({ra['n']} val states, both sides uniformly random)"))
    term = {"val": terminal_breakdown(predictions(model, data["val"]), env_cfg, tol)}
    if "heldout_opp" in data:
        term["heldout_opp"] = terminal_breakdown(predictions(model, data["heldout_opp"]), env_cfg, tol)
    term["random_actions"] = terminal_breakdown(
        random_action_predictions(model, data["val"], env_cfg, int(ra["n"]), int(ra["seed"])), env_cfg, tol)
    write_csv(out / "terminal_reward.csv", [{"set": k, **r} for k, v in term.items() for r in v])
    tables.append(format_terminal_table(term))
    write_csv(out / "errors.csv", rows)
    write_csv(out / "errors_by_group.csv", group_rows)
    text = "\n\n".join(tables)
    (out / "error_table.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nwrote {out}/errors.csv, errors_by_group.csv, error_table.md")


if __name__ == "__main__":
    main()

"""Offline training of the GRU predictor on collected data (collects it first if missing).

python -m scripts.train_predictor --config configs/predictor.yaml
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.logging_utils import save_run_metadata
from adaptive_npc.models.player_model import build_model, save_predictor
from adaptive_npc.models.predictor_data import collect_split, load_windows
from adaptive_npc.seeding import seed_everything


@torch.no_grad()
def evaluate(model, data, batch=2048):
    model.eval()
    loss = correct = 0.0
    for s in range(0, len(data), batch):
        x, y = data.batch(np.arange(s, min(s + batch, len(data))))
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
    return loss / len(data), correct / len(data)


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg["seed"])
    d, t, window = cfg["data"], cfg["train"], int(cfg["model"]["window"])
    data_dir = Path(d["dir"])
    if not (data_dir / "train").exists():
        env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
        collect_split(cfg, env_cfg, players_cfg, d["rounds_per_archetype"], d["train_seed_base"], data_dir / "train")
        collect_split(cfg, env_cfg, players_cfg, d["val_rounds_per_archetype"], d["val_seed_base"], data_dir / "val")
    train, val = load_windows(data_dir / "train", window), load_windows(data_dir / "val", window)

    model = build_model(cfg["model"])
    opt = torch.optim.Adam(model.parameters(), lr=float(t["lr"]))
    gen = np.random.default_rng(cfg["seed"])
    out = Path(args.out or cfg["eval"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch in range(1, int(t["epochs"]) + 1):
        model.train()
        perm = gen.permutation(len(train))
        for s in range(0, len(perm), int(t["batch_size"])):
            x, y = train.batch(perm[s:s + int(t["batch_size"])])
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        tl, ta = evaluate(model, train)
        vl, va = evaluate(model, val)
        rows.append({"epoch": epoch, "train_nll": tl, "train_acc": ta, "val_nll": vl, "val_acc": va})
        print(f"epoch {epoch:3d} train nll {tl:.4f} acc {ta:.3f} | val nll {vl:.4f} acc {va:.3f}", flush=True)

    ckpt = Path(t["checkpoint"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_predictor(ckpt, model, cfg["model"])
    with open(out / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_run_metadata(out, cfg)
    print(f"saved {ckpt} and {out / 'train_log.csv'}")


if __name__ == "__main__":
    main()

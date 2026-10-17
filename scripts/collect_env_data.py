"""Collect (state, NPC action, opponent action, next state, reward) transitions for the env model Γ.

python -m scripts.collect_env_data --config configs/env_model.yaml [--recollect]

Every split (train / val / heldout_opp) plays every NPC (scripted, random, ppo_none / ppo_predicted / ppo_oracle
checkpoints, stochastic acting) against every opponent of that split for `rounds` rounds. Opponents: the archetypes,
random, the online-learning naive / reasoning learners and the frozen exploiters. train and val use the same
opponents and NPCs but different env seeds; heldout_opp uses opponents that never appear in train.
Saved to <data.dir>/<split>.npz (gitignored) and reused unless the collect config changed or --recollect.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from adaptive_npc.config import add_config_args, config_from_args, load_config
from adaptive_npc.env.gym_env import NPCEnv
from adaptive_npc.eval.runner import build_npc
from adaptive_npc.models.env_model import raw_state
from adaptive_npc.npc.ppo_npc import PolicyPlayer
from adaptive_npc.players.base import RandomPlayer
from adaptive_npc.players.scripted import make_player
from scripts.check_learner_drift import make_learner


def npc_cells(col):
    """[(method, checkpoint_seed or None)]: scripted / random once, each ppo mode per checkpoint seed."""
    cells = []
    for m in col["npc_methods"]:
        cells += [(m, int(s)) for s in col["npc_checkpoint_seeds"]] if m.startswith("ppo_") else [(m, None)]
    return cells


def make_opponent(name, col, env_cfg, players_cfg, learners_cfg, seed, cell_idx):
    if name == "random":
        return RandomPlayer(seed)
    if name in ("naive_learner", "reasoning_learner"):  # learning online during the cell, fresh from pretrained weights
        entry = {"lr": learners_cfg["ppo_lite"]["lr"], "update_every": learners_cfg["update_every"]}
        lseeds = col["learner_seeds"]
        return make_learner(learners_cfg, name.split("_")[0], int(lseeds[cell_idx % len(lseeds)]), entry, True)
    if name.startswith("exploiter_seed"):
        return PolicyPlayer(f"{col['exploiter_dir']}/{name}.pt", deterministic=False, seed=seed)
    return make_player(name, players_cfg, env_cfg, seed)


def collect_split(cfg, env_cfg, players_cfg, learners_cfg, split):
    col, sp = cfg["collect"], cfg["collect"]["splits"][split]
    opps = list(col["opponents"][sp["opponents"]])
    npc_cfg = {"eval": {"deterministic": bool(col["npc_deterministic"]), "predictor_online": False}, "online": None,
               "predictor_checkpoint": col["predictor_checkpoint"], "train": {"model_dir": col["npc_model_dir"]}}
    npcs = npc_cells(col)
    npc_names = list(col["npc_methods"])
    keys = ("state", "next_state", "a_npc", "a_opp", "reward", "done", "ep", "npc_id", "opp_id")
    buf = {k: [] for k in keys}
    ep, cell = 0, 0
    for ni, (method, ckpt) in enumerate(npcs):
        for oi, opp_name in enumerate(opps):
            seed = int(sp["env_seed_base"]) + cell
            opp = make_opponent(opp_name, col, env_cfg, players_cfg, learners_cfg, seed + 1, cell)
            base = NPCEnv(env_cfg, opponent=opp, seed=seed)
            env, act_fn, on_rs = build_npc(npc_cfg, env_cfg, players_cfg, method, ckpt, base, seed + 2)
            game = base.game
            npc_id = npc_names.index(method)
            for rnd in range(int(sp["rounds"])):
                obs, _ = env.reset(seed=seed if rnd == 0 else None)
                if on_rs:
                    on_rs()
                hist, done = [], False
                while not done:
                    s = raw_state(game)
                    a = int(act_fn(obs, hist))
                    obs, r, done, _, info = env.step(a)
                    hist.append((a, info["opp_action"]))
                    for k, v in zip(keys, (s, raw_state(game), a, info["opp_action"], r, done, ep, npc_id, oi)):
                        buf[k].append(v)
                ep += 1
            cell += 1
    data = {"state": np.array(buf["state"], np.int16), "next_state": np.array(buf["next_state"], np.int16),
            "a_npc": np.array(buf["a_npc"], np.int8), "a_opp": np.array(buf["a_opp"], np.int8),
            "reward": np.array(buf["reward"], np.float32), "done": np.array(buf["done"], bool),
            "ep": np.array(buf["ep"], np.int32), "npc_id": np.array(buf["npc_id"], np.int8),
            "opp_id": np.array(buf["opp_id"], np.int8), "npc_names": np.array(npc_names), "opp_names": np.array(opps)}
    return data


def collect_config_hash(cfg, env_cfg):
    blob = json.dumps({"collect": cfg["collect"], "env": env_cfg}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


def load_or_collect(cfg, env_cfg, players_cfg, learners_cfg, split, recollect=False):
    path = Path(cfg["data"]["dir"]) / f"{split}.npz"
    h = collect_config_hash(cfg, env_cfg)
    if path.exists() and not recollect:
        with np.load(path) as z:
            if str(z["cfg_hash"]) == h:
                return {k: z[k] for k in z.files if k != "cfg_hash"}
        print(f"[{split}] collect config changed -> recollecting", flush=True)
    data = collect_split(cfg, env_cfg, players_cfg, learners_cfg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cfg_hash=h, **data)
    print(f"[{split}] {len(data['state'])} transitions, {int(data['ep'].max()) + 1} rounds -> {path}", flush=True)
    return data


def load_all(cfg, recollect=False):
    torch.set_num_threads(1)
    env_cfg, players_cfg = load_config(cfg["env_config"]), load_config(cfg["players_config"])
    learners_cfg = load_config(cfg["learners_config"])
    return {s: load_or_collect(cfg, env_cfg, players_cfg, learners_cfg, s, recollect) for s in cfg["collect"]["splits"]}


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--recollect", action="store_true", help="ignore cached .npz files")
    args = parser.parse_args()
    load_all(config_from_args(args), args.recollect)


if __name__ == "__main__":
    main()

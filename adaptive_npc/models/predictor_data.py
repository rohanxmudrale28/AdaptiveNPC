"""Collect (obs, npc action, player action) trajectories and load them as WindowData."""
from pathlib import Path

import numpy as np

from adaptive_npc.env.match import MatchHook, run_match
from adaptive_npc.env.observations import encode
from adaptive_npc.models.player_model import WindowData
from adaptive_npc.npc.scripted_npc import make_scripted_npc
from adaptive_npc.players.base import RandomPlayer
from adaptive_npc.players.scripted import make_player


def make_npc(name, players_cfg, env_cfg, seed=0):
    if name == "random":
        return RandomPlayer(seed)
    if name == "scripted_npc":
        return make_scripted_npc(players_cfg, env_cfg, seed)
    raise ValueError(f"unknown NPC kind {name!r}")


class Recorder(MatchHook):
    """Records NPC-view obs before every step plus both actions."""

    def __init__(self):
        self.obs, self.a_npc, self.a_pl, self.round_start = [], [], [], []
        self._cur = None

    def on_round_start(self, round_idx, game, npc, player):
        self._cur = encode(game, 0)
        self._first = True

    def on_step(self, round_idx, game, a_npc, a_player):
        self.obs.append(self._cur)
        self.a_npc.append(a_npc)
        self.a_pl.append(a_player)
        self.round_start.append(self._first)
        self._first = False
        self._cur = encode(game, 0)

    def arrays(self):
        return {"obs": np.array(self.obs, dtype=np.float32), "a_npc": np.array(self.a_npc, dtype=np.int64),
                "a_pl": np.array(self.a_pl, dtype=np.int64), "round_start": np.array(self.round_start, dtype=bool)}


def collect_match(env_cfg, players_cfg, npc_name, archetype, n_rounds, seed):
    npc = make_npc(npc_name, players_cfg, env_cfg, seed)
    player = make_player(archetype, players_cfg, env_cfg, seed + 1)
    rec = Recorder()
    run_match(env_cfg, npc, player, n_rounds=n_rounds, seed=seed, hook=rec)
    return rec.arrays()


def collect_split(cfg, env_cfg, players_cfg, rounds, seed_base, out_dir):
    """One .npz per (npc, archetype). Returns the list of written paths."""
    archetypes = cfg["data"]["archetypes"] or players_cfg["train"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ni, npc_name in enumerate(cfg["data"]["npcs"]):
        for ai, arch in enumerate(archetypes):
            arrays = collect_match(env_cfg, players_cfg, npc_name, arch, rounds, seed_base + 100 * ni + ai)
            path = out / f"{npc_name}__{arch}.npz"
            np.savez_compressed(path, **arrays)
            paths.append(path)
    return paths


def load_windows(data_dir, window):
    """Concatenate every .npz in data_dir into one WindowData (round boundaries preserved)."""
    parts = [np.load(p) for p in sorted(Path(data_dir).glob("*.npz"))]
    if not parts:
        raise FileNotFoundError(f"no .npz files in {data_dir}; run scripts.collect_data first")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("obs", "a_npc", "a_pl", "round_start")}
    return WindowData(cat["obs"], cat["a_npc"], cat["a_pl"], cat["round_start"], window)

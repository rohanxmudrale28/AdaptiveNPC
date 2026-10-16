"""Hand-coded baseline NPC (a Player, so it plugs into match.py / NPCEnv like any other)."""
from adaptive_npc.players.scripted import ScriptedNPC, make_player


def make_scripted_npc(players_cfg, env_cfg, seed=None):
    return make_player("scripted_npc", players_cfg, env_cfg, seed)


__all__ = ["ScriptedNPC", "make_scripted_npc"]

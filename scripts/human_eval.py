"""Run one human-evaluation session (M13): practice vs the Training Dummy, then 2 matches of 10 rounds against each
of two blind opponents (A and B = the scripted NPC and ppo_predicted, order counterbalanced by participant ID).

python -m scripts.human_eval --participant P01
python -m scripts.human_eval --participant P02 --no-rating
python -m scripts.human_eval --participant P01 --dry-run        # print the plan only, no window
python -m scripts.human_eval --participant P01 --music          # sound effects are always on; music is off unless asked
Logs: results/human/<ID>_session.json (plan, ratings, per-match logs). Only the ID is stored, no personal data.
"""
import argparse
import re
import sys
import time
import zlib
from pathlib import Path

from adaptive_npc.config import apply_overrides, load_config
from adaptive_npc.ui.session import write_json

ID_RE = r"[A-Za-z0-9_-]{1,32}"


def npc_order(participant, npcs):
    """NPC list ordered [A, B]. Counterbalanced by ID: the trailing number's parity (P01 -> first NPC as A,
    P02 -> second), or a CRC of the ID if it has no trailing number."""
    m = re.search(r"(\d+)$", participant)
    k = int(m.group(1)) if m else zlib.crc32(participant.encode())
    return list(npcs) if k % 2 == 1 else list(reversed(npcs))


def build_plan(participant, cfg):
    """List of match dicts in play order: practice first, then blocks of matches_per_npc per labelled opponent."""
    order = npc_order(participant, cfg["npcs"])
    plan = [{"phase": "practice", "label": cfg["practice"]["label"], "opponent": cfg["practice"]["opponent"],
             "rounds": cfg["practice"]["rounds"], "seed": cfg["seed"]}]
    n = 0
    for label, method in zip(cfg["labels"], order):
        for rep in range(cfg["matches_per_npc"]):
            n += 1
            plan.append({"phase": "test", "label": label, "opponent": method, "rounds": cfg["rounds_per_match"],
                         "seed": cfg["seed"] + n, "match_in_block": rep})
    return plan


def ask_rating(input_fn=input):
    """1-5 difficulty rating from the console. Enter (or anything invalid after one retry) skips."""
    for _ in range(2):
        txt = input_fn("Difficulty of that opponent, 1 (very easy) to 5 (very hard), Enter to skip: ").strip()
        if not txt:
            return None
        if txt in {"1", "2", "3", "4", "5"}:
            return int(txt)
        print("please type a number from 1 to 5, or press Enter")
    return None


def run_match(entry, participant, args, play_main):
    """Run one blind match through scripts/play.py. Returns its summary (with 'path'), or None if nothing was played."""
    argv = ["--blind", "--opponent", entry["opponent"], "--label", entry["label"], "--participant", participant,
            "--rounds", str(entry["rounds"]), "--seed", str(entry["seed"]), "--set", f"ui.human_dir={args.out_dir}/matches"]
    if args.tick_rate is not None:
        argv += ["--tick-rate", str(args.tick_rate)]
    argv += ["--music"] if args.music else ["--no-music"]
    if args.mute:
        argv += ["--mute"]
    return play_main(argv)


def main(argv=None, play_main=None, input_fn=input):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participant", required=True, help="participant ID (letters, digits, _ or -; no names)")
    p.add_argument("--config", default="configs/human_eval.yaml")
    p.add_argument("--tick-rate", type=float, default=None, help="override render.yaml tick_rate")
    p.add_argument("--no-rating", action="store_true", help="skip the difficulty-rating prompt")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--music", action="store_true", help="play music too (default: off; effects are the same for every opponent)")
    p.add_argument("--mute", action="store_true", help="start with all sound muted")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    if not re.fullmatch(ID_RE, args.participant):
        sys.exit("--participant must be 1-32 characters of letters, digits, '_' or '-' (an ID, not a name)")
    cfg = apply_overrides(load_config(args.config), args.overrides)
    args.out_dir = (args.out_dir or cfg["out_dir"]).rstrip("/\\")
    plan = build_plan(args.participant, cfg)
    tick = args.tick_rate if args.tick_rate is not None else load_config("configs/render.yaml")["tick_rate"]
    order = npc_order(args.participant, cfg["npcs"])
    print(f"participant {args.participant}: Opponent A = {order[0]}, Opponent B = {order[1]}, tick rate {tick}/s")
    if args.dry_run:
        for i, m in enumerate(plan):
            print(f"  {i}: {m['phase']:8s} Opponent {m['label']:8s} {m['opponent']:14s} {m['rounds']} rounds seed {m['seed']}")
        return None
    if play_main is None:
        from scripts import play
        play_main = play.main
    out = Path(args.out_dir) / f"{args.participant}_session.json"
    session = {"participant": args.participant, "tick_rate": tick, "order": {"A": order[0], "B": order[1]},
               "config": cfg, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "matches": []}
    ask = cfg["ask_rating"] and not args.no_rating
    for i, entry in enumerate(plan):
        label = "practice (Training Dummy)" if entry["phase"] == "practice" else f"Opponent {entry['label']}"
        print(f"\n--- match {i + 1}/{len(plan)}: {label}, {entry['rounds']} rounds. Press Enter to start.")
        input_fn("")
        summary = run_match(entry, args.participant, args, play_main)
        rec = {k: entry[k] for k in ("phase", "label", "opponent", "rounds", "seed")}
        rec["completed"] = bool(summary and summary.get("completed"))
        rec["summary"] = summary
        if entry["phase"] == "test" and rec["completed"] and ask:
            rec["difficulty_rating"] = ask_rating(input_fn)
        session["matches"].append(rec)
        write_json(out, session)  # after every match so an interrupted session keeps its data
        if not rec["completed"]:
            print("match not completed: session stopped. Re-run with the same ID to start over (the new file replaces this one).")
            break
    else:
        session["finished"] = True
        write_json(out, session)
        print(f"\nsession complete. Thank you! Saved {out}")
    return session


if __name__ == "__main__":
    main()

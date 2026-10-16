"""Run bookkeeping: copy the resolved config and git hash next to every result."""
import subprocess
from pathlib import Path

import yaml


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_run_metadata(out_dir, cfg):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    (out / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")

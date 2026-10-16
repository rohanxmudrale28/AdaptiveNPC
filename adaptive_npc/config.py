"""YAML config loading with dotted CLI overrides (e.g. --set ppo.lr=1e-3)."""
import argparse
import copy
from pathlib import Path

import yaml


def load_config(path, overrides=None):
    """Load a YAML file into a dict, then apply 'a.b.c=value' overrides.

    Override values are parsed as YAML, so '3' -> int, '1e-3' -> float,
    'true' -> bool, '[0,1]' -> list.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides or [])


def _parse_value(raw):
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # PyYAML (YAML 1.1) reads '3e-4' as a string; recover floats.
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value = _parse_value(raw)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {key!r}: {p!r} is not a mapping")
        node[parts[-1]] = value
    return cfg


def add_config_args(parser):
    """Add the standard --config / --seed / --out / --set flags to a script parser."""
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="overrides config seed")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="dotted config override (repeatable)")
    return parser


def config_from_args(args):
    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def parse_args(argv=None, description=None):
    parser = add_config_args(argparse.ArgumentParser(description=description))
    return parser.parse_args(argv)

"""Command line entry points: make_data, train, evaluate, compare, export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values, e.g. lora.r=16")


def make_data_main(argv=None) -> None:
    from .config import load_config
    from .data import generate_splits, write_jsonl

    p = argparse.ArgumentParser(description="Generate the synthetic train/val/test/hard splits.")
    _common(p)
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    d = cfg["data"]
    splits = generate_splits(d["seed"], d["n_train"], d["n_val"], d["n_test"], d["n_hard"])
    for name, rows in splits.items():
        write_jsonl(rows, Path(d["dir"]) / f"{name}.jsonl")
        print(f"{name}: {len(rows)} rows")


def train_main(argv=None) -> None:
    from .config import load_config, run_name
    from .train import train

    p = argparse.ArgumentParser(description="Train a LoRA adapter.")
    _common(p)
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    cfg["_run_name"] = run_name(cfg)
    meta = train(cfg)
    print(json.dumps({k: meta[k] for k in ("run", "trainable_params", "total_params", "trainable_pct", "train_seconds")}))


def evaluate_main(argv=None) -> None:
    from .config import load_config
    from .evaluate import METHODS, evaluate

    p = argparse.ArgumentParser(description="Evaluate one method on the test and hard sets.")
    _common(p)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--retry", action="store_true", help="validate JSON against the schema and retry once")
    p.add_argument("--label", default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    label = a.label or (a.method + ("_retry" if a.retry else ""))
    evaluate(cfg, a.method, a.adapter, a.retry, label)


def compare_main(argv=None) -> None:
    from .compare import compare

    p = argparse.ArgumentParser(description="Build results/RESULTS.md from results/runs.")
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results/RESULTS.md")
    a = p.parse_args(argv)
    print(compare(a.runs, a.out))


def export_main(argv=None) -> None:
    from .config import load_config
    from .export import merge_and_export

    p = argparse.ArgumentParser(description="Merge a LoRA adapter into the base weights and save.")
    _common(p)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", default="outputs/merged")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    merge_and_export(cfg["model"]["name"], a.adapter, a.out)


if __name__ == "__main__":
    import sys

    cmds = {"make_data": make_data_main, "train": train_main, "evaluate": evaluate_main,
            "compare": compare_main, "export": export_main}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: python -m lorafin.cli {{{','.join(cmds)}}} [args]")
    cmds[sys.argv[1]](sys.argv[2:])

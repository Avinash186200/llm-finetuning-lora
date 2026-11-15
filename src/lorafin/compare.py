"""Collect results/runs/*.json into one markdown report."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .data import FIELDS


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def load_runs(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]


def mean_std(xs: list[float]) -> str:
    if len(xs) == 1:
        return f"{pct(xs[0])}"
    return f"{pct(statistics.mean(xs))} ± {pct(statistics.stdev(xs))}"


def build_report(runs: list[dict]) -> str:
    if not runs:
        return "No runs found.\n"
    env = runs[0]["env"]
    hw = env["gpu"] or f"CPU ({env['threads']} threads)"
    lines = [f"# Results", "", f"Model: `{runs[0]['model']}`. Hardware: {hw}. torch {env['torch']}. "
             "Greedy decoding. Data seed 1234. Latency is batch wall time divided by batch size.", ""]

    lines += ["## Main comparison (percent)", "",
              "| Run | Method | Retry | Test valid | Test exact | Hard valid | Hard exact | ms/example (test) | Prompt tok | Gen tok |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        t, h = r["splits"]["test"], r["splits"]["hard"]
        lines.append(f"| {r['label']} | {r['method']} | {'yes' if r['retry'] else 'no'} | {pct(t['json_valid'])} | {pct(t['exact_match'])} | "
                     f"{pct(h['json_valid'])} | {pct(h['exact_match'])} | {t['latency_ms_per_example']} | {t['prompt_tokens_mean']} | {t['gen_tokens_mean']} |")

    for split in ("test", "hard"):
        lines += ["", f"## Per-field accuracy, {split} set (percent)", "",
                  "| Run | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS)]
        for r in runs:
            s = r["splits"][split]
            lines.append(f"| {r['label']} | " + " | ".join(pct(s[f'acc_{k}']) for k in FIELDS) + " |")

    lora = [r for r in runs if r.get("train_meta") and not r["retry"]]
    if lora:
        lines += ["", "## LoRA runs", "",
                  "| Run | r | alpha | lr | seed | Trainable | Total | Trainable % | Train s | Final val loss | Test exact | Hard exact |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in lora:
            m = r["train_meta"]
            lines.append(f"| {r['label']} | {m['lora']['r']} | {m['lora']['alpha']} | {m['train']['lr']} | {m['train']['seed']} | "
                         f"{m['trainable_params']:,} | {m['total_params']:,} | {m['trainable_pct']} | {m['train_seconds']} | "
                         f"{m['history'][-1]['val_loss']} | {pct(r['splits']['test']['exact_match'])} | {pct(r['splits']['hard']['exact_match'])} |")
        by_r: dict[int, list[dict]] = {}
        for r in lora:
            by_r.setdefault(r["train_meta"]["lora"]["r"], []).append(r)
        lines += ["", "## LoRA by rank (mean ± std over seeds, percent)", "",
                  "| r | Seeds | Test exact | Hard exact | Hard valid |", "|---|---|---|---|---|"]
        for rank in sorted(by_r):
            rs = by_r[rank]
            seeds = ",".join(str(x["train_meta"]["train"]["seed"]) for x in rs)
            lines.append(f"| {rank} | {seeds} | {mean_std([x['splits']['test']['exact_match'] for x in rs])} | "
                         f"{mean_std([x['splits']['hard']['exact_match'] for x in rs])} | {mean_std([x['splits']['hard']['json_valid'] for x in rs])} |")

    probed = [r for r in runs if r.get("probes")]
    if probed:
        lines += ["", "## Knowledge probes (percent)", "",
                  "| Run | Recall of training values (n=30) | World facts (n=10) |", "|---|---|---|"]
        for r in probed:
            p = r["probes"]
            lines.append(f"| {r['label']} | {pct(p['recall_seen_training_values']['recall_acc'])} | {pct(p['world_facts']['world_acc'])} |")
    return "\n".join(lines) + "\n"


def compare(run_dir: str = "results/runs", out: str = "results/RESULTS.md") -> str:
    report = build_report(load_runs(Path(run_dir)))
    Path(out).write_text(report, encoding="utf-8")
    return report

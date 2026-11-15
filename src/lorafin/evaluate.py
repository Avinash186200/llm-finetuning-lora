"""Evaluate one method on the test and hard sets, plus the knowledge probes."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import modeling
from .data import read_jsonl
from .prompts import messages, retry_messages
from .scoring import aggregate, extract_json, schema_errors, score_one

METHODS = ["zero_shot", "few_shot", "lora"]

# General facts the training data never touches. Used to check that the
# tuned model did not lose what the base model already knew.
WORLD_PROBES = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("How many days are in a week?", ["7", "seven"]),
    ("What color is the sky on a clear day?", ["blue"]),
    ("What is the largest planet in the solar system?", ["jupiter"]),
    ("What is 2 plus 2?", ["4", "four"]),
    ("What gas do plants absorb from the air?", ["carbon dioxide", "co2"]),
    ("Which ocean lies between Africa and Australia?", ["indian"]),
    ("What is the freezing point of water in Celsius?", ["0", "zero"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
]


def _shots(data_dir: Path, n: int) -> list[dict]:
    return read_jsonl(data_dir / "train.jsonl")[:n]


def run_extraction(model, tok, rows, shots, cfg, retry: bool) -> dict:
    bs, mnt = cfg["eval"]["batch_size"], cfg["eval"]["max_new_tokens"]
    preds, n_prompt, n_gen, secs, retried = [], [], [], 0.0, 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        texts, npt, ngt, s = modeling.generate_batch(model, tok, [messages(r["text"], shots) for r in chunk], mnt)
        secs += s
        if retry:
            bad = [j for j, t in enumerate(texts) if schema_errors(extract_json(t))]
            if bad:
                retried += len(bad)
                rmsgs = [retry_messages(chunk[j]["text"], texts[j], schema_errors(extract_json(texts[j])), shots) for j in bad]
                t2, npt2, ngt2, s2 = modeling.generate_batch(model, tok, rmsgs, mnt)
                secs += s2
                for k, j in enumerate(bad):
                    texts[j] = t2[k]
                    npt[j] += npt2[k]
                    ngt[j] += ngt2[k]
        preds += texts
        n_prompt += npt
        n_gen += ngt
    scores = [score_one(p, r["target"]) for p, r in zip(preds, rows)]
    m = aggregate(scores)
    m.update({
        "latency_ms_per_example": round(1000 * secs / len(rows), 1),
        "prompt_tokens_mean": round(sum(n_prompt) / len(rows), 1),
        "gen_tokens_mean": round(sum(n_gen) / len(rows), 1),
        "retried": retried,
    })
    return {"metrics": m, "predictions": [{"text": r["text"], "pred": p, **s} for r, p, s in zip(rows, preds, scores)]}


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def run_recall_probe(model, tok, rows, cfg) -> dict:
    """Ask for values the model saw during training. Fine-tuning on
    extraction should not turn those values into recallable facts."""
    qs = [[{"role": "user", "content": f"What was the {r['target']['metric']} of {r['target']['entity']} in "
            f"{r['target']['period']}? Answer with the number only."}] for r in rows]
    texts, *_ = modeling.generate_batch(model, tok, qs, 16)
    hits = 0
    for t, r in zip(texts, rows):
        nums = [float(x.replace(",", "")) for x in _NUM.findall(t) if x.replace(",", "").replace(".", "").lstrip("-")]
        hits += any(abs(n - r["target"]["value"]) < 1e-6 for n in nums)
    return {"n": len(rows), "recall_acc": hits / len(rows), "samples": texts[:5]}


def run_world_probe(model, tok) -> dict:
    qs = [[{"role": "user", "content": q + " Answer in a few words."}] for q, _ in WORLD_PROBES]
    texts, *_ = modeling.generate_batch(model, tok, qs, 16)
    hits = sum(any(a in t.lower() for a in ans) for t, (_, ans) in zip(texts, WORLD_PROBES))
    return {"n": len(WORLD_PROBES), "world_acc": hits / len(WORLD_PROBES), "answers": texts}


def evaluate(cfg: dict, method: str, adapter: str | None, retry: bool, label: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "lora" and not adapter:
        raise ValueError("method lora needs --adapter")
    modeling.configure_threads(cfg["train"]["threads"])
    modeling.set_seed(0)
    data_dir = Path(cfg["data"]["dir"])
    tok = modeling.load_tokenizer(cfg["model"]["name"])
    model = modeling.load_for_eval(cfg["model"]["name"], adapter if method == "lora" else None,
                                   cfg["model"]["quantize_4bit_on_cuda"])
    shots = _shots(data_dir, cfg["eval"]["n_few_shot"]) if method == "few_shot" else None
    result = {"label": label, "method": method, "retry": retry, "adapter": adapter,
              "model": cfg["model"]["name"], "env": modeling.device_info(), "splits": {}}
    pred_dir = Path(cfg["output_dir"]) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "hard"):
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        r = run_extraction(model, tok, rows, shots, cfg, retry)
        result["splits"][split] = r["metrics"]
        (pred_dir / f"{label}_{split}.json").write_text(json.dumps(r["predictions"], indent=1))
        print(label, split, json.dumps(r["metrics"]), flush=True)
    if not retry and method in ("zero_shot", "lora"):
        train_rows = read_jsonl(data_dir / "train.jsonl")[50:80]
        result["probes"] = {"recall_seen_training_values": run_recall_probe(model, tok, train_rows, cfg),
                            "world_facts": run_world_probe(model, tok)}
        print(label, "probes", json.dumps({k: {kk: vv for kk, vv in v.items() if kk.endswith("acc")} for k, v in result["probes"].items()}), flush=True)
    if adapter:
        meta = Path(adapter) / "train_meta.json"
        if meta.exists():
            result["train_meta"] = json.loads(meta.read_text())
    out = Path("results") / "runs" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result

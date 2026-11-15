from lorafin.compare import build_report, mean_std


def _run(label, method, exact, retry=False, meta=None, probes=None):
    split = {"n": 10, "json_valid": 1.0, "exact_match": exact, "latency_ms_per_example": 12.0,
             "prompt_tokens_mean": 100.0, "gen_tokens_mean": 40.0, "retried": 0,
             **{f"acc_{k}": exact for k in ("entity", "metric", "value", "unit", "period", "direction")}}
    r = {"label": label, "method": method, "retry": retry, "model": "m", "splits": {"test": split, "hard": split},
         "env": {"gpu": None, "threads": 4, "torch": "x"}}
    if meta:
        r["train_meta"] = meta
    if probes:
        r["probes"] = probes
    return r


def _meta(r, seed):
    return {"lora": {"r": r, "alpha": 2 * r}, "train": {"lr": 5e-4, "seed": seed}, "trainable_params": 1000,
            "total_params": 100000, "trainable_pct": 1.0, "train_seconds": 3.0, "history": [{"val_loss": 0.1}]}


def test_mean_std():
    assert mean_std([0.5]) == "50.0"
    assert mean_std([0.4, 0.6]) == "50.0 ± 14.1"


def test_report_tables():
    probes = {"recall_seen_training_values": {"recall_acc": 0.0}, "world_facts": {"world_acc": 0.5}}
    runs = [_run("zero_shot", "zero_shot", 0.1, probes=probes),
            _run("lora_r8_s0", "lora", 0.9, meta=_meta(8, 0)),
            _run("lora_r8_s1", "lora", 0.7, meta=_meta(8, 1))]
    rep = build_report(runs)
    assert "| zero_shot | zero_shot | no | 100.0 | 10.0 |" in rep
    assert "| 8 | 0,1 | 80.0 ± 14.1 |" in rep
    assert "| zero_shot | 0.0 | 50.0 |" in rep
    assert "CPU (4 threads)" in rep


def test_empty_report():
    assert build_report([]) == "No runs found.\n"

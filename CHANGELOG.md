# Changelog

## 1.0.0 - 2026-11-15

- Seeded synthetic generator for sentence-to-JSON extraction (train, val, test, hard).
- Zero-shot, few-shot and LoRA comparison with JSON validity, exact match, per-field accuracy, latency and token counts.
- Schema validation with one retry for base and tuned models.
- Rank ablation (r = 4, 8, 16) and three seeds at r = 8.
- Knowledge probes: recall of training values and general facts.
- QLoRA (4-bit) path when CUDA is present. Not benchmarked here.
- Merge-and-export script.
- pytest suite and Ubuntu CI that needs no model download.

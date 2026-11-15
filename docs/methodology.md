# Methodology

This page says exactly what I measured and how, so you can judge the numbers in `results/RESULTS.md`.

## Task

Input: one short, made-up financial news sentence.
Output: one JSON object with six keys.

| Key | Type | Example |
|---|---|---|
| entity | string | "Halvorn Labs" |
| metric | one of six strings | "revenue" |
| value | number, as written in the sentence | 4.2 |
| unit | million, billion, percent, usd, employees | "billion" |
| period | "Q3 2024" or "FY2024" | "Q3 2024" |
| direction | up, down, flat | "up" |

I picked this task because it is narrow and because a script can score it. There is no judge model and no human rating.

## Data

`src/lorafin/data.py` builds every split from one seed (1234).

- Company names are random syllables plus a suffix ("Zepex Dynamics"). No real firm appears.
- train 600, val 100, test 200, hard 80.
- Val, test and hard use only company names that never appear in train. So the model cannot score by remembering a name.
- Test uses the same sentence templates as train. It measures "did it learn the format".
- Hard uses templates the model never saw, plus distractors: a prior-period value ("climbed from $41 million to $52 million"), a rival company, and unseen period forms ("3Q22", "the 2023 third quarter", "fiscal year 2024"). It measures "does it generalise past the templates".

The data is synthetic and template-based. That is the main limitation. Real news is messier.

## Methods

All methods use the same model, the same instruction text (`src/lorafin/prompts.py`) and greedy decoding with a 128-token cap.

1. **Zero-shot.** Instruction plus the sentence, through the model's chat template.
2. **Few-shot.** The same, with three solved training examples as earlier chat turns.
3. **LoRA.** Adapters trained on the zero-shot prompt format. Loss is only on the answer tokens. At test time the prompt is identical to zero-shot, so the only difference is the adapter.
4. **Retry.** Optional wrapper for any method. If the output fails the schema check, I send the bad output back with the error list and ask once more. This is validation plus retry, not token-level constrained decoding. I did not implement a grammar-constrained decoder.

## LoRA settings

| Setting | Value |
|---|---|
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Rank r | 8 (ablation: 4 and 16) |
| Alpha | 2 x r (so the scale alpha/r stays at 2) |
| Dropout | 0.05 |
| Learning rate | 5e-4, linear warmup 10% then linear decay |
| Optimiser | AdamW, no weight decay, grad clip 1.0 |
| Batch size | 8 |
| Epochs | 2 (150 steps) |
| Max length | 320 tokens (no example is truncated; the code raises if one would be) |
| Seeds | 0, 1, 2 at r = 8; seed 0 for r = 4 and r = 16 |

I chose these once, before seeing test numbers, and did not tune them on test. Val loss is logged per epoch only as a sanity check.

## Metrics

- **JSON valid:** the output contains a JSON object that passes the schema check (all six keys, no extra keys, right types, metric and direction from the allowed lists).
- **Exact match:** all six fields equal the target. Strings compare after lower-casing and collapsing whitespace. Numbers compare after rounding to 4 decimals.
- **Per-field accuracy:** share of examples where that one field is right. An invalid output counts as wrong on every field.
- **Latency:** wall time of `model.generate` for a batch of 16, divided by 16. It includes the retry call when one happens. It is a CPU number with batching; do not read it as single-request latency.
- **Prompt / generated tokens:** mean per example. Generated tokens count up to the first end-of-sequence token.
- **Trainable parameters:** parameters with `requires_grad` after PEFT wraps the model, against the total.
- **Training time:** wall time of the training loop only, not model loading.

## Knowledge probes

Two small checks run on the zero-shot base model and on each LoRA model.

1. **Recall of training values (n = 30).** For 30 training sentences, I ask "What was the revenue of X in Q3 2024? Answer with the number only." The model saw each of these values during training. If fine-tuning taught facts, it should answer them. A hit is any number in the reply that equals the value.
2. **General facts (n = 10).** Ten easy questions ("capital of France"). The training data has nothing like them. A drop here would be a sign of catastrophic forgetting.

Both probes are tiny. They are a sanity check, not a study.

## Environment

The machine has no usable GPU: `torch.cuda.is_available()` returns False. All results are CPU float32. The 4-bit QLoRA path in `modeling.load_base` runs only when CUDA is present. I did not run it, so I report no numbers for it.

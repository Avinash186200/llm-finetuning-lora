# Walkthrough

This is the path I would take a new engineer through. Each step is one command and one idea. Run them from the repo root after the quick start in the README.

## 1. Look at the data first

```bash
python -m lorafin.cli make_data
head -n 2 data/train.jsonl
head -n 2 data/hard.jsonl
```

Each line is a sentence and its target JSON. Train and test come from the same templates. Hard uses new phrasing and adds a distractor: an old value, a rival company, or a period written as "3Q22". Before any model work, read ten examples of each. Most fine-tuning bugs I have seen were data bugs.

What to notice: the period is normalised ("the third quarter of 2024" becomes "Q3 2024"). The model has to learn that mapping. A prompt can describe it, but a 135M model does not follow descriptions well.

## 2. Measure the base model with no tuning

```bash
python -m lorafin.cli evaluate --method zero_shot
python -m lorafin.cli evaluate --method few_shot
```

Always do this first. If prompting already gets you there, you do not need to train anything. Open `outputs/predictions/zero_shot_test.json` and read the raw outputs, not just the score. With this small model you will see it copy the instruction text into the fields ("metric": "one of: revenue, net income, ...") and pretty-print JSON until it hits the token cap.

## 3. Train a LoRA adapter

```bash
python -m lorafin.cli train --set lora.r=8 lora.alpha=16 train.seed=0
```

What happens in `src/lorafin/train.py`:

1. Load the base model and freeze it.
2. `get_peft_model` adds two small matrices A (r x d) and B (d x r) next to each attention projection. Only these train. B starts at zero, so step 0 is exactly the base model.
3. Each example is the zero-shot prompt plus the target JSON plus the end token. Labels are -100 on the prompt, so loss is only on the answer.
4. A plain AdamW loop with warmup and linear decay. No `Trainer`, so you can read every line.
5. The adapter (a few MB) is saved to `outputs/adapters/lora_r8_s0/`. The base weights are not copied.

The log prints the trainable share. Here it is 0.68% of the model at r = 8.

A bug I hit here, worth knowing. My first run used `max_len: 192`. The prompt is about 161 tokens and the answer about 50, so every example was cut before the closing brace and the end token. Val loss still looked great (0.006), because the loss only covers tokens that are present. But the model never learned to stop: it wrote the six right fields, then kept inventing keys ("country", "symbol") until the cap. Test JSON validity was 22%. After raising `max_len` to 320 the same run scored 100%. `encode_example` now raises if any example would be truncated. Low loss is not proof that training worked; read the outputs.

## 4. Evaluate the adapter on the same prompt

```bash
python -m lorafin.cli evaluate --method lora --adapter outputs/adapters/lora_r8_s0 --label lora_r8_s0
```

The prompt is identical to zero-shot. The only change is the adapter. That makes the comparison fair.

## 5. Add validation and one retry

```bash
python -m lorafin.cli evaluate --method few_shot --retry
python -m lorafin.cli evaluate --method lora --retry --adapter outputs/adapters/lora_r8_s0 --label lora_r8_s0_retry
```

If the output fails the schema check, the error list goes back to the model and it tries once more. This is cheap insurance in production. It cannot fix a model that does not know the format, and it does nothing for outputs that are valid JSON but wrong.

## 6. Check what the adapter did not learn

The evaluate step for zero-shot and LoRA also runs two probes:

- Recall: "What was the revenue of X in Q3 2024?" for values the model trained on.
- World facts: ten easy general questions.

Compare base and tuned. See `results/RESULTS.md` for what I got.

## 7. Merge and export

```bash
python -m lorafin.cli export --adapter outputs/adapters/lora_r8_s0 --out outputs/merged_r8_s0
```

`merge_and_unload` folds B x A x (alpha / r) into the frozen weight. The result is a normal model with no PEFT dependency and no extra latency. Keep the adapter if you want to swap tasks on one base model; merge if you want one artifact to serve.

## 8. Compare everything

```bash
python -m lorafin.cli compare
```

This writes `results/RESULTS.md` from `results/runs/*.json`. I never edit those tables by hand.

## On a GPU

With CUDA present and `model.quantize_4bit_on_cuda: true`, `load_base` loads the base in 4-bit NF4 with bitsandbytes and calls `prepare_model_for_kbit_training`. That is QLoRA. The adapters still train in higher precision. Install with `pip install -e ".[train,gpu]"`. I have not run this path, so I make no claim about its numbers.

## Exercises

1. Drop `k_proj` and `o_proj` from `lora.target_modules`. How much do trainable parameters fall, and does hard-set accuracy move?
2. Train for 1 epoch instead of 2. Where does the drop show up first: validity or exact match?
3. Add a new hard template with a currency other than dollars. Which method breaks?
4. Swap the model for `Qwen/Qwen2.5-0.5B-Instruct` with `--set model.name=...`. Re-run zero-shot. Is the fine-tuning gap smaller?

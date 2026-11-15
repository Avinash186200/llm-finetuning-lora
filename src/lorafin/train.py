"""LoRA fine-tuning with a plain PyTorch loop (no Trainer, so every step is visible)."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from . import modeling
from .data import read_jsonl
from .prompts import messages, target_text


def encode_example(tok, ex: dict, max_len: int) -> dict:
    """Tokenise prompt + answer; loss only on the answer tokens."""
    prompt = tok.apply_chat_template(messages(ex["text"]), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target_text(ex["target"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_len:
        # Truncating would cut the closing brace and end token, and the
        # model would never learn to stop. Fail loudly instead.
        raise ValueError(f"example is {len(p_ids) + len(a_ids)} tokens, over max_len={max_len}")
    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids
    return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def eval_loss(model, batches) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in batches:
        b = {k: v.to(model.device) for k, v in b.items()}
        tot += model(**b).loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)


def train(cfg: dict) -> dict:
    from peft import LoraConfig, get_peft_model

    tcfg, lcfg = cfg["train"], cfg["lora"]
    modeling.configure_threads(tcfg["threads"])
    modeling.set_seed(tcfg["seed"])
    data_dir = Path(cfg["data"]["dir"])
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")

    tok = modeling.load_tokenizer(cfg["model"]["name"])
    model = modeling.load_base(cfg["model"]["name"], cfg["model"]["quantize_4bit_on_cuda"], for_training=True)
    lora = LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                      target_modules=list(lcfg["target_modules"]), task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable, total = modeling.count_params(model)

    enc_train = [encode_example(tok, ex, tcfg["max_len"]) for ex in train_rows]
    enc_val = [encode_example(tok, ex, tcfg["max_len"]) for ex in val_rows]
    bs = tcfg["batch_size"]
    val_batches = [collate(enc_val[i:i + bs], tok.pad_token_id) for i in range(0, len(enc_val), bs)]

    steps_per_epoch = math.ceil(len(enc_train) / bs)
    total_steps = steps_per_epoch * tcfg["epochs"]
    warmup = max(1, int(total_steps * tcfg["warmup_ratio"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))

    rng = random.Random(tcfg["seed"])
    history = []
    model.train()
    t0 = time.perf_counter()
    step = 0
    for epoch in range(tcfg["epochs"]):
        order = list(range(len(enc_train)))
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            b = collate([enc_train[j] for j in order[i:i + bs]], tok.pad_token_id)
            b = {k: v.to(model.device) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
        vl = eval_loss(model, val_batches)
        history.append({"epoch": epoch + 1, "step": step, "train_loss_last": round(loss.item(), 4), "val_loss": round(vl, 4)})
        print(f"epoch {epoch + 1} val_loss {vl:.4f}", flush=True)
    train_secs = time.perf_counter() - t0

    out_dir = Path(cfg["output_dir"]) / "adapters" / cfg["_run_name"]
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    meta = {
        "run": cfg["_run_name"], "model": cfg["model"]["name"], "lora": lcfg,
        "train": tcfg, "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4), "train_seconds": round(train_secs, 1),
        "steps": total_steps, "history": history, "env": modeling.device_info(),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta

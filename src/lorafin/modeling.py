"""Model loading, seeding and batched greedy generation."""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_info() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base(name: str, quantize_4bit_on_cuda: bool = True, for_training: bool = False):
    """Load the base model. On CUDA with 4-bit enabled this is the QLoRA path."""
    from transformers import AutoModelForCausalLM

    if torch.cuda.is_available() and quantize_4bit_on_cuda:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
        if for_training:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        return model
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def load_for_eval(name: str, adapter_dir: str | None = None, quantize_4bit_on_cuda: bool = True):
    model = load_base(name, quantize_4bit_on_cuda)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def configure_threads(n: int) -> None:
    n = min(n, os.cpu_count() or n)
    torch.set_num_threads(n)


@torch.no_grad()
def generate_batch(model, tok, list_of_messages: list[list[dict]], max_new_tokens: int) -> tuple[list[str], list[int], list[int], float]:
    """Greedy-decode a batch. Returns texts, prompt token counts, generated token counts, seconds."""
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in list_of_messages]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    secs = time.perf_counter() - t0
    gen = out[:, enc["input_ids"].shape[1]:]
    texts, n_gen = [], []
    for row in gen:
        ids = row.tolist()
        # Count up to and including the first EOS; the rest is padding.
        if tok.eos_token_id in ids:
            ids = ids[: ids.index(tok.eos_token_id) + 1]
        n_gen.append(len(ids))
        texts.append(tok.decode(ids, skip_special_tokens=True))
    n_prompt = enc["attention_mask"].sum(dim=1).tolist()
    return texts, n_prompt, n_gen, secs


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

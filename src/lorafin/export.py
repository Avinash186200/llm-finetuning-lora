"""Merge a LoRA adapter into full-precision base weights.

Merging always loads the base in full precision, even on a GPU: you
cannot merge into 4-bit weights without a lossy round trip.
"""
from __future__ import annotations

from pathlib import Path


def merge_and_export(base_name: str, adapter_dir: str, out_dir: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out = Path(out_dir)
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    print(f"merged model written to {out}")
    return out

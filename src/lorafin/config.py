"""YAML config loading with dot-path overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "model": {"name": "HuggingFaceTB/SmolLM2-135M-Instruct", "quantize_4bit_on_cuda": True},
    "data": {"seed": 1234, "dir": "data", "n_train": 600, "n_val": 100, "n_test": 200, "n_hard": 80},
    "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    "train": {"seed": 0, "epochs": 2, "batch_size": 8, "lr": 5.0e-4, "weight_decay": 0.0,
              "warmup_ratio": 0.1, "max_len": 320, "threads": 16},
    "eval": {"batch_size": 16, "max_new_tokens": 128, "n_few_shot": 3},
    "output_dir": "outputs",
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in out:
            raise ConfigError(f"unknown config key: {k}")
        if isinstance(out[k], dict):
            if not isinstance(v, dict):
                raise ConfigError(f"{k} must be a mapping")
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(s: str):
    v = yaml.safe_load(s)
    if isinstance(v, str):  # YAML 1.1 reads "1e-4" as a string
        try:
            return float(v)
        except ValueError:
            pass
    return v


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b=value`` overrides (value parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key=value: {item}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise ConfigError(f"unknown config key: {key}")
            node = node[p]
        if parts[-1] not in node:
            raise ConfigError(f"unknown config key: {key}")
        node[parts[-1]] = _parse_scalar(val)
    return cfg


def validate(cfg: dict) -> dict:
    lo = cfg["lora"]
    if int(lo["r"]) <= 0 or float(lo["alpha"]) <= 0:
        raise ConfigError("lora.r and lora.alpha must be positive")
    if not lo["target_modules"]:
        raise ConfigError("lora.target_modules must not be empty")
    if float(cfg["train"]["lr"]) <= 0:
        raise ConfigError("train.lr must be positive")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return validate(apply_overrides(cfg, overrides or []))


def run_name(cfg: dict) -> str:
    return f"lora_r{cfg['lora']['r']}_s{cfg['train']['seed']}"

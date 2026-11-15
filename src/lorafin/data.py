"""Seeded synthetic generator for the extraction task.

Each example is one short financial-news sentence and the strict JSON
record it should map to. Everything is made up: company names are
built from random syllables, so no real firm appears in the data.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

FIELDS = ["entity", "metric", "value", "unit", "period", "direction"]
METRICS = ["revenue", "net income", "operating margin", "earnings per share", "free cash flow", "headcount"]
UNITS = {
    "revenue": ["million", "billion"],
    "net income": ["million", "billion"],
    "free cash flow": ["million", "billion"],
    "operating margin": ["percent"],
    "earnings per share": ["usd"],
    "headcount": ["employees"],
}
DIRECTIONS = ["up", "down", "flat"]

_SYL_A = ["Hal", "Ver", "Cor", "Bel", "Tam", "Ost", "Ner", "Lum", "Kav", "Pri", "Sol", "Dra", "Mer", "Quin", "Zep", "Arl"]
_SYL_B = ["vorn", "ex", "ida", "ton", "ara", "ix", "elle", "ondo", "ari", "usk", "ent", "oria", "ade", "ulon"]
_SUFFIX = ["Dynamics", "Holdings", "Labs", "Industries", "Group", "Systems", "Foods", "Energy", "Logistics", "Bank"]

_QWORD = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def make_entity(rng: random.Random) -> str:
    return f"{rng.choice(_SYL_A)}{rng.choice(_SYL_B)} {rng.choice(_SUFFIX)}"


def _value(rng: random.Random, unit: str) -> float:
    if unit == "percent":
        return round(rng.uniform(2, 45), 1)
    if unit == "usd":
        return round(rng.uniform(0.1, 9.9), 2)
    if unit == "employees":
        return float(rng.randrange(200, 90000, 50))
    if unit == "billion":
        return round(rng.uniform(1, 80), 1)
    return float(rng.randrange(20, 990))


def _prior_value(rng: random.Random, v: float, unit: str, direction: str) -> float:
    """A distractor number that is consistent with the stated direction."""
    if direction == "up":
        f = rng.uniform(0.6, 0.95)
    elif direction == "down":
        f = rng.uniform(1.05, 1.5)
    else:
        f = rng.choice([rng.uniform(0.6, 0.95), rng.uniform(1.05, 1.5)])
    x = v * f
    if unit in ("million", "employees"):
        return float(max(1, round(x / 10) * 10 if unit == "employees" else round(x)))
    if unit == "usd":
        return round(x, 2)
    return round(x, 1)


def _fmt_value(v: float, unit: str) -> str:
    if unit == "percent":
        return f"{v}%"
    if unit == "usd":
        return f"${v:.2f}"
    if unit == "employees":
        return f"{int(v):,}"
    num = f"{int(v)}" if v == int(v) else f"{v}"
    return f"${num} {unit}"


def _period(rng: random.Random, hard: bool) -> tuple[str, str]:
    """Return (canonical period, surface text)."""
    year = rng.randint(2019, 2026)
    if rng.random() < 0.25:
        canon = f"FY{year}"
        forms = [f"fiscal {year}", f"FY{year}", f"full-year {year}"]
        if hard:
            forms = [f"the {year} fiscal year", f"fiscal year {year}"]
        return canon, rng.choice(forms)
    q = rng.randint(1, 4)
    canon = f"Q{q} {year}"
    forms = [f"Q{q} {year}", f"the {_QWORD[q]} quarter of {year}"]
    if hard:
        forms = [f"{q}Q{str(year)[2:]}", f"the {year} {_QWORD[q]} quarter"]
    return canon, rng.choice(forms)


_EASY = {
    "up": [
        "{e} reported {m} of {v} for {p}, up from a year earlier.",
        "In {p}, {e} said {m} rose to {v}.",
        "{e}'s {m} increased to {v} in {p}.",
    ],
    "down": [
        "{e} reported {m} of {v} for {p}, down from a year earlier.",
        "In {p}, {e} said {m} fell to {v}.",
        "{e}'s {m} declined to {v} in {p}.",
    ],
    "flat": [
        "{e} reported {m} of {v} for {p}, unchanged from a year earlier.",
        "In {p}, {e} said {m} held steady at {v}.",
    ],
}

# Hard templates: unseen paraphrases plus distractors (a prior value, a
# rival company, an unrelated second number).
_HARD = {
    "up": [
        "Shares of {e} jumped after {m} climbed from {old} to {v} in {p}, beating rival {r}.",
        "{m_cap} at {e} came in at {v} for {p}; analysts at {r} had pencilled in {old}, and the figure was higher than last year.",
        "For {p}, {r} lagged while {e} grew its {m} to {v}.",
    ],
    "down": [
        "{e} saw {m} slip from {old} to {v} in {p}, while {r} held up better.",
        "{m_cap} at {e} came in at {v} for {p}, lower than the {old} posted a year before.",
        "Despite {r} raising guidance, {e}'s {m} dropped to {v} in {p}.",
    ],
    "flat": [
        "{e} kept {m} essentially level at {v} in {p}, compared with {old} at {r}.",
        "{m_cap} at {e} was {v} for {p}, no change on the prior year, unlike {r}.",
    ],
}


def make_example(rng: random.Random, hard: bool = False) -> dict:
    entity = make_entity(rng)
    metric = rng.choice(METRICS)
    unit = rng.choice(UNITS[metric])
    value = _value(rng, unit)
    direction = rng.choice(DIRECTIONS)
    period, p_text = _period(rng, hard)
    fields = dict(e=entity, m=metric, m_cap=metric.capitalize(), v=_fmt_value(value, unit), p=p_text)
    if hard:
        rival = make_entity(rng)
        while rival == entity:
            rival = make_entity(rng)
        fields.update(r=rival, old=_fmt_value(_prior_value(rng, value, unit, direction), unit))
        tmpl = rng.choice(_HARD[direction])
    else:
        tmpl = rng.choice(_EASY[direction])
    text = tmpl.format(**fields)
    target = {"entity": entity, "metric": metric, "value": value, "unit": unit, "period": period, "direction": direction}
    return {"text": text, "target": target}


def generate_splits(seed: int, n_train: int, n_val: int, n_test: int, n_hard: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "hard": []}
    seen_entities: set[str] = set()
    # Train first, then keep held-out entities disjoint from train.
    for name, n, hard in [("train", n_train, False), ("val", n_val, False), ("test", n_test, False), ("hard", n_hard, True)]:
        while len(splits[name]) < n:
            ex = make_example(rng, hard=hard)
            ent = ex["target"]["entity"]
            if name == "train":
                seen_entities.add(ent)
            elif ent in seen_entities:
                continue
            splits[name].append(ex)
    return splits


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

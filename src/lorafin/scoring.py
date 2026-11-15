"""Parse model output into JSON and score it against the target record."""
from __future__ import annotations

import json
import re

from .data import DIRECTIONS, FIELDS, METRICS

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict | None:
    """Return the first JSON object in ``text``, or None.

    Accepts a bare object or one wrapped in a markdown code fence. Takes
    the first balanced ``{...}`` span so trailing chatter is ignored.
    """
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def schema_errors(obj: dict | None) -> list[str]:
    """Check the record against the task schema. Empty list means valid."""
    if obj is None:
        return ["output is not a JSON object"]
    errs = []
    missing = [k for k in FIELDS if k not in obj]
    extra = [k for k in obj if k not in FIELDS]
    if missing:
        errs.append(f"missing keys: {missing}")
    if extra:
        errs.append(f"unexpected keys: {extra}")
    if "value" in obj and (isinstance(obj["value"], bool) or not isinstance(obj["value"], (int, float))):
        errs.append("value must be a number")
    for k in ("entity", "metric", "unit", "period", "direction"):
        if k in obj and not isinstance(obj[k], str):
            errs.append(f"{k} must be a string")
    if isinstance(obj.get("metric"), str) and obj["metric"] not in METRICS:
        errs.append(f"metric must be one of {METRICS}")
    if isinstance(obj.get("direction"), str) and obj["direction"] not in DIRECTIONS:
        errs.append(f"direction must be one of {DIRECTIONS}")
    return errs


def _norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    if isinstance(v, str):
        return " ".join(v.strip().lower().split())
    return v


def field_matches(pred: dict | None, target: dict) -> dict[str, bool]:
    if pred is None:
        return {k: False for k in FIELDS}
    return {k: k in pred and _norm(pred[k]) == _norm(target[k]) for k in FIELDS}


def score_one(raw: str, target: dict) -> dict:
    obj = extract_json(raw)
    errs = schema_errors(obj)
    fm = field_matches(obj, target)
    return {"valid": not errs, "exact": all(fm.values()), "fields": fm, "errors": errs}


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "json_valid": sum(s["valid"] for s in scores) / n,
        "exact_match": sum(s["exact"] for s in scores) / n,
    }
    for k in FIELDS:
        out[f"acc_{k}"] = sum(s["fields"][k] for s in scores) / n
    return out

"""Prompt construction. The same instruction is used for every method."""
from __future__ import annotations

import json

from .data import DIRECTIONS, METRICS

INSTRUCTION = (
    "Extract one record from the sentence. Reply with JSON only, with exactly these keys: "
    "entity (company name), metric (one of: " + ", ".join(METRICS) + "), "
    "value (number as written, no symbols), unit (million, billion, percent, usd or employees), "
    "period (like \"Q3 2024\" or \"FY2024\"), direction (one of: " + ", ".join(DIRECTIONS) + ")."
)


def target_text(target: dict) -> str:
    return json.dumps(target)


def messages(text: str, shots: list[dict] | None = None) -> list[dict]:
    msgs = []
    for s in shots or []:
        msgs.append({"role": "user", "content": f"{INSTRUCTION}\nSentence: {s['text']}"})
        msgs.append({"role": "assistant", "content": target_text(s["target"])})
    msgs.append({"role": "user", "content": f"{INSTRUCTION}\nSentence: {text}"})
    return msgs


def retry_messages(text: str, bad_output: str, errors: list[str], shots: list[dict] | None = None) -> list[dict]:
    msgs = messages(text, shots)
    msgs.append({"role": "assistant", "content": bad_output})
    msgs.append({"role": "user", "content": "That reply was invalid: " + "; ".join(errors) + ". Reply again with the corrected JSON only."})
    return msgs

from __future__ import annotations

import json
import re
from typing import Any

from .base import Evaluator
from .types import EvalItem


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_json(text: str) -> Any | None:
    candidate = _strip_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None


def _word_count(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def score_constraints(response: str, constraints: dict[str, Any]) -> float:
    kind = constraints.get("kind")
    text = response.strip()
    if kind == "json":
        payload = _parse_json(text)
        if not isinstance(payload, dict):
            return 0.0
        required = list(constraints.get("required_keys") or [])
        if any(key not in payload for key in required):
            return 0.0
        equals = constraints.get("equals") or {}
        if any(payload.get(key) != value for key, value in equals.items()):
            return 0.0
        array_len = constraints.get("array_len") or {}
        for key, length in array_len.items():
            value = payload.get(key)
            if not isinstance(value, list) or len(value) != length:
                return 0.0
        return 1.0
    if kind == "regex":
        pattern = constraints["pattern"]
        return float(bool(re.fullmatch(pattern, text, flags=re.MULTILINE)))
    if kind == "bullet_list":
        prefix = constraints.get("prefix", "-")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        matched = [line for line in lines if line.startswith(prefix)]
        return float(len(matched) == int(constraints["count"]) and len(matched) == len(lines))
    if kind == "numbered_list":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        expected = int(constraints["count"])
        ok = len(lines) == expected and all(
            line.startswith(f"{index}.") for index, line in enumerate(lines, start=1)
        )
        return float(ok)
    if kind == "prefix_max_words":
        prefix = str(constraints["prefix"])
        if not text.startswith(prefix):
            return 0.0
        if _word_count(text) > int(constraints["max_words"]):
            return 0.0
        forbidden = [str(token).lower() for token in constraints.get("forbidden") or []]
        lowered = text.lower()
        if any(token in lowered for token in forbidden):
            return 0.0
        return 1.0
    if kind == "contains_max_words":
        if _word_count(text) > int(constraints["max_words"]):
            return 0.0
        needles = [str(token) for token in constraints.get("needles") or []]
        return float(all(needle in text for needle in needles))
    return 0.0


class InstructionFollowing(Evaluator):
    name = "instruction_following"
    description = "Format, JSON schema, and constraint checks on compact instruction items."
    dataset_file = "instruction_following.jsonl"

    def parse_row(self, row: dict[str, Any]) -> EvalItem:
        return EvalItem(
            id=str(row["id"]),
            prompt=str(row["prompt"]),
            gold=str(row.get("gold", "")),
            constraints=dict(row.get("constraints") or {}),
        )

    def score(self, item: EvalItem, response: str) -> float:
        return score_constraints(response, item.constraints)

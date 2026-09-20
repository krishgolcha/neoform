from __future__ import annotations

from .arithmetic import ArithmeticExactMatch
from .base import Evaluator
from .gsm8k import GSM8KExactMatch
from .instruction import InstructionFollowing

_REGISTRY: dict[str, Evaluator] = {
    evaluator.name: evaluator
    for evaluator in (ArithmeticExactMatch(), GSM8KExactMatch(), InstructionFollowing())
}


def available_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_evaluator(name: str) -> Evaluator:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(available_names())
        raise ValueError(f"Unknown evaluator {name!r}. Available: {available}") from exc


def list_evaluators() -> list[dict[str, object]]:
    return [
        {
            "name": evaluator.name,
            "description": evaluator.description,
            "examples": evaluator.catalog_size(),
        }
        for evaluator in _REGISTRY.values()
    ]

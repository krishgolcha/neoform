from __future__ import annotations

from typing import Any

from .base import Evaluator
from .numbers import extract_first_number, numbers_equal
from .types import EvalItem


class ArithmeticExactMatch(Evaluator):
    name = "arithmetic_exact_match"
    description = (
        "Exact numeric match on bundled arithmetic items (dataset file, not hardcoded tuples)."
    )
    dataset_file = "arithmetic.jsonl"

    def parse_row(self, row: dict[str, Any]) -> EvalItem:
        return EvalItem(id=str(row["id"]), prompt=str(row["prompt"]), gold=str(row["gold"]))

    def score(self, item: EvalItem, response: str) -> float:
        predicted = extract_first_number(response)
        return float(predicted is not None and numbers_equal(predicted, item.gold))

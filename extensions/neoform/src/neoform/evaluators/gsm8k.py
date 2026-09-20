from __future__ import annotations

from typing import Any

from .base import Evaluator
from .numbers import extract_final_number, numbers_equal
from .types import EvalItem


class GSM8KExactMatch(Evaluator):
    name = "gsm8k_exact_match"
    description = (
        "Compact GSM8K-style word problems scored by extracting the final number "
        "(prefers a #### answer marker)."
    )
    dataset_file = "gsm8k.jsonl"

    def parse_row(self, row: dict[str, Any]) -> EvalItem:
        return EvalItem(
            id=str(row["id"]),
            prompt=str(row["prompt"]),
            gold=str(row["gold"]),
            metadata={"rationale": row.get("rationale", "")},
        )

    def score(self, item: EvalItem, response: str) -> float:
        predicted = extract_final_number(response)
        return float(predicted is not None and numbers_equal(predicted, item.gold))

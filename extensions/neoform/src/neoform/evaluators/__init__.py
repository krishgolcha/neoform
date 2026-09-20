"""Pluggable evaluators keyed by BenchmarkSpec.name."""

from .registry import available_names, get_evaluator, list_evaluators
from .runner import run_evaluations, training_pairs_for_spec
from .types import EvalItem

__all__ = [
    "EvalItem",
    "available_names",
    "get_evaluator",
    "list_evaluators",
    "run_evaluations",
    "training_pairs_for_spec",
]

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from importlib import resources
from typing import Any

from neoform.domain import BenchmarkSpec

from .types import EvalItem


def load_jsonl(filename: str) -> list[dict[str, Any]]:
    text = resources.files("neoform.datasets").joinpath(filename).read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def select_items(pool: list[EvalItem], spec: BenchmarkSpec, seed: int) -> list[EvalItem]:
    if spec.examples > len(pool):
        raise ValueError(
            f"{spec.name} requested {spec.examples} examples but only {len(pool)} are bundled"
        )
    rng = random.Random(f"{spec.name}:{seed}:{spec.examples}")
    chosen = pool[:]
    rng.shuffle(chosen)
    return chosen[: spec.examples]


class Evaluator(ABC):
    name: str
    description: str
    dataset_file: str

    def catalog_size(self) -> int:
        return len(load_jsonl(self.dataset_file))

    def all_items(self) -> list[EvalItem]:
        return [self.parse_row(row) for row in load_jsonl(self.dataset_file)]

    def load_items(self, spec: BenchmarkSpec, seed: int) -> list[EvalItem]:
        return select_items(self.all_items(), spec, seed)

    def training_pairs(self) -> list[tuple[str, str]]:
        return [(item.prompt, item.reference_output()) for item in self.all_items()]

    @abstractmethod
    def parse_row(self, row: dict[str, Any]) -> EvalItem: ...

    @abstractmethod
    def score(self, item: EvalItem, response: str) -> float: ...

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from .domain import CostEstimate, EvolutionSpec, Usage

MODELS_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"


def _price(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", value)
    if not match:
        raise ValueError(f"Invalid Tinker price: {value!r}")
    return float(match.group())


@dataclass(frozen=True)
class ModelPrice:
    tinker_id: str
    train_per_million: float
    sample_per_million: float
    prefill_per_million: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.train_tokens * self.train_per_million
            + usage.sample_tokens * self.sample_per_million
            + usage.prompt_tokens * self.prefill_per_million
        ) / 1_000_000


class PricingCatalog:
    def __init__(self, prices: dict[str, ModelPrice], source: str = MODELS_URL):
        self.prices = prices
        self.source = source

    @classmethod
    def from_payload(cls, payload: list[dict[str, object]], source: str = MODELS_URL):
        prices = {}
        for item in payload:
            model_id = str(item["tinker_id"])
            prices[model_id] = ModelPrice(
                tinker_id=model_id,
                train_per_million=_price(item["train"]),
                sample_per_million=_price(item["sample"]),
                prefill_per_million=_price(item["prefill"]),
            )
        return cls(prices, source)

    @classmethod
    async def fetch(cls, cache_path: Path | None = None) -> PricingCatalog:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(MODELS_URL)
                response.raise_for_status()
                payload = response.json()
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, indent=2))
            return cls.from_payload(payload)
        except (httpx.HTTPError, ValueError, KeyError):
            if cache_path and cache_path.exists():
                return cls.from_payload(json.loads(cache_path.read_text()), str(cache_path))
            raise

    def get(self, model_id: str) -> ModelPrice:
        try:
            return self.prices[model_id]
        except KeyError as exc:
            raise ValueError(f"No pricing found for {model_id}") from exc

    def estimate(self, spec: EvolutionSpec) -> CostEstimate:
        price = self.get(spec.base_model)
        candidates = spec.search.population * spec.search.generations
        train_tokens = (
            candidates
            * spec.search.steps_per_candidate
            * spec.search.batch_size
            * spec.search.max_sequence_tokens
        )
        eval_examples = sum(item.examples for item in spec.benchmarks)
        max_out = max(spec.mutations.max_tokens)
        prompt_tokens = candidates * eval_examples * 256
        sample_tokens = candidates * eval_examples * max_out
        train_usd = train_tokens * price.train_per_million / 1_000_000
        sample_usd = (
            prompt_tokens * price.prefill_per_million
            + sample_tokens * price.sample_per_million
        ) / 1_000_000
        return CostEstimate(
            train_usd=round(train_usd, 4),
            sample_usd=round(sample_usd, 4),
            reserved_usd=round((train_usd + sample_usd) * 1.1, 4),
            pricing_source=self.source,
        )


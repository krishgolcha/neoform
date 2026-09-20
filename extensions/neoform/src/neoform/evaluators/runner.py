from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from neoform.domain import EvaluationResult, EvolutionSpec, Genotype, Usage

from .registry import get_evaluator
from .types import EvalItem

CompleteFn = Callable[[EvalItem], Awaitable[tuple[str, Usage]]]


async def run_evaluations(
    spec: EvolutionSpec,
    genotype: Genotype,
    complete: CompleteFn,
) -> EvaluationResult:
    scores: dict[str, float] = {}
    examples: list[dict[str, object]] = []
    usage = Usage()
    started = time.perf_counter()
    scored = 0
    for benchmark in spec.benchmarks:
        evaluator = get_evaluator(benchmark.name)
        items = evaluator.load_items(spec=benchmark, seed=genotype.curriculum_seed)
        rewards: list[float] = []
        for item in items:
            text, item_usage = await complete(item)
            reward = evaluator.score(item, text)
            rewards.append(reward)
            scored += 1
            usage = Usage(
                train_tokens=usage.train_tokens + item_usage.train_tokens,
                sample_tokens=usage.sample_tokens + item_usage.sample_tokens,
                prompt_tokens=usage.prompt_tokens + item_usage.prompt_tokens,
            )
            examples.append(
                {
                    "benchmark": benchmark.name,
                    "id": item.id,
                    "prompt": item.prompt,
                    "response": text,
                    "reward": reward,
                }
            )
        scores[benchmark.name] = sum(rewards) / len(rewards) if rewards else 0.0
    elapsed_ms = (time.perf_counter() - started) * 1000
    return EvaluationResult(
        scores=scores,
        latency_ms=elapsed_ms / scored if scored else elapsed_ms,
        usage=usage,
        examples=examples,
    )


def training_pairs_for_spec(spec: EvolutionSpec) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for benchmark in spec.benchmarks:
        pairs.extend(get_evaluator(benchmark.name).training_pairs())
    if not pairs:
        raise ValueError("No training pairs available for the selected benchmarks")
    return pairs

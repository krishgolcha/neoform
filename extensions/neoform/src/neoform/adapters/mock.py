from __future__ import annotations

import asyncio
import hashlib
import random
import time

from neoform.domain import EvaluationResult, EvolutionSpec, Genotype, TrainingResult, Usage

from .base import LabAdapter


def _seed(*parts: object) -> int:
    value = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(value).hexdigest()[:12], 16)


class MockTinkerAdapter(LabAdapter):
    """Deterministic test double. It is never selected by the production CLI by default."""

    async def doctor(self) -> dict[str, object]:
        return {"connected": True, "provider": "mock", "capabilities": ["train", "sample"]}

    async def train(
        self,
        candidate_id: str,
        spec: EvolutionSpec,
        genotype: Genotype,
        parent_state: str | None,
    ) -> TrainingResult:
        await asyncio.sleep(0)
        rng = random.Random(_seed(candidate_id, genotype.model_dump_json(), parent_state))
        tokens = spec.search.steps_per_candidate * spec.search.batch_size * 384
        return TrainingResult(
            state_checkpoint=f"tinker://mock/{candidate_id}/weights/state",
            sampler_checkpoint=f"tinker://mock/{candidate_id}/sampler_weights/model",
            training_loss=round(0.45 + rng.random() * 0.8, 4),
            usage=Usage(train_tokens=tokens),
        )

    async def probe(self, sampler_checkpoint: str) -> bool:
        await asyncio.sleep(0)
        return "invalid" not in sampler_checkpoint

    async def evaluate(
        self, sampler_checkpoint: str, spec: EvolutionSpec, genotype: Genotype
    ) -> EvaluationResult:
        started = time.perf_counter()
        rng = random.Random(_seed(sampler_checkpoint, genotype.model_dump_json()))
        lr_center = 8e-5
        lr_bonus = max(0.0, 0.08 - abs(genotype.learning_rate - lr_center) * 500)
        scores = {
            benchmark.name: round(min(0.98, 0.52 + lr_bonus + rng.random() * 0.22), 4)
            for benchmark in spec.benchmarks
        }
        examples = [
            {
                "prompt": "If a crew installs 12 panels per hour for 7 hours, how many panels?",
                "response": "84 panels",
                "reward": 1,
            }
        ]
        total = sum(item.examples for item in spec.benchmarks)
        return EvaluationResult(
            scores=scores,
            latency_ms=max(1.0, (time.perf_counter() - started) * 1000),
            usage=Usage(prompt_tokens=total * 128, sample_tokens=total * 192),
            examples=examples,
        )

    async def chat(
        self, sampler_checkpoint: str, messages: list[dict[str, str]], **parameters: object
    ) -> dict[str, object]:
        return {
            "id": "chatcmpl_mock",
            "object": "chat.completion",
            "model": sampler_checkpoint,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "NEOFORM mock response for automated testing.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }


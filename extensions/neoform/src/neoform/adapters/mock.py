from __future__ import annotations

import asyncio
import hashlib
import random

from neoform.domain import EvaluationResult, EvolutionSpec, Genotype, TrainingResult, Usage
from neoform.evaluators import EvalItem, run_evaluations, training_pairs_for_spec

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
        training_pairs_for_spec(spec)
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
        rng = random.Random(_seed(sampler_checkpoint, genotype.model_dump_json()))
        lr_center = 8e-5
        lr_bonus = max(0.0, 0.08 - abs(genotype.learning_rate - lr_center) * 500)
        threshold = min(0.92, 0.48 + lr_bonus)

        async def complete(item: EvalItem) -> tuple[str, Usage]:
            await asyncio.sleep(0)
            text = item.reference_output() if rng.random() < threshold else "I do not know."
            prompt_tokens = max(8, len(item.prompt.split()) * 2)
            sample_tokens = max(4, len(text.split()) * 2)
            return text, Usage(prompt_tokens=prompt_tokens, sample_tokens=sample_tokens)

        return await run_evaluations(spec, genotype, complete)

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

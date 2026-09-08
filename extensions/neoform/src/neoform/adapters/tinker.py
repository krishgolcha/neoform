from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from neoform.domain import EvaluationResult, EvolutionSpec, Genotype, TrainingResult, Usage

from .base import LabAdapter


class TinkerAdapter(LabAdapter):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TINKER_API_KEY")

    def _require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("TINKER_API_KEY is required for live NEOFORM runs")
        return self.api_key

    def _service(self, spec: EvolutionSpec | None = None):
        import tinker

        kwargs: dict[str, Any] = {"api_key": self._require_key()}
        if spec and spec.project_id:
            kwargs["project_id"] = spec.project_id
        return tinker.ServiceClient(**kwargs)

    async def doctor(self) -> dict[str, object]:
        service = self._service()
        capabilities = await service.get_server_capabilities_async()
        return {
            "connected": True,
            "provider": "tinker",
            "capabilities": capabilities.model_dump(mode="json"),
        }

    @staticmethod
    def _examples() -> list[tuple[str, str]]:
        return [
            ("Calculate 17 * 23. Give only the answer.", "391"),
            ("A box has 9 rows of 14 bolts. How many bolts?", "126"),
            ("What is 144 divided by 12?", "12"),
            ("A crew works 8 hours at 15 units per hour. Total?", "120"),
        ]

    async def train(
        self,
        candidate_id: str,
        spec: EvolutionSpec,
        genotype: Genotype,
        parent_state: str | None,
    ) -> TrainingResult:
        if genotype.loss_function != "cross_entropy":
            raise ValueError(
                "The live Tinker adapter currently supports only cross_entropy training"
            )

        from tinker import types

        service = self._service(spec)
        metadata = {"neoform_candidate": candidate_id, "neoform_evolution": spec.name}
        if parent_state:
            create = (
                service.create_training_client_from_state_with_optimizer_async
                if genotype.optimizer_mode == "resume"
                else service.create_training_client_from_state_async
            )
            training = await create(parent_state, user_metadata=metadata)
        else:
            training = await service.create_lora_training_client_async(
                base_model=spec.base_model,
                rank=spec.search.rank,
                seed=genotype.curriculum_seed,
                user_metadata=metadata,
            )

        tokenizer = training.get_tokenizer()
        examples = self._examples()
        total_tokens = 0
        losses: list[float] = []
        for step in range(spec.search.steps_per_candidate):
            batch = []
            for offset in range(spec.search.batch_size):
                example_index = (step + offset + genotype.curriculum_seed) % len(examples)
                prompt, completion = examples[example_index]
                prompt_text = f"User: {prompt}\nAssistant:"
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
                completion_tokens = tokenizer.encode(f" {completion}", add_special_tokens=False)
                full = prompt_tokens + completion_tokens
                total_tokens += len(full)
                batch.append(
                    types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=full[:-1]),
                        loss_fn_inputs={
                            "weights": [0.0] * (len(prompt_tokens) - 1)
                            + [1.0] * len(completion_tokens),
                            "target_tokens": full[1:],
                        },
                    )
                )
            future = await training.forward_backward_async(batch, genotype.loss_function)
            result = await future.result_async()
            loss = getattr(result, "loss", None)
            if loss is not None:
                losses.append(float(loss))
            optim = await training.optim_step_async(
                types.AdamParams(learning_rate=genotype.learning_rate)
            )
            await optim.result_async()

        ttl = spec.checkpoint_ttl_days * 86_400
        state_future = await training.save_state_async(
            f"neoform-{candidate_id}-state", ttl_seconds=ttl, overwrite=True
        )
        state = await state_future.result_async()
        sampler_future = await training.save_weights_for_sampler_async(
            f"neoform-{candidate_id}-sampler", ttl_seconds=ttl
        )
        sampler = await sampler_future.result_async()
        return TrainingResult(
            state_checkpoint=state.path,
            sampler_checkpoint=sampler.path,
            training_loss=sum(losses) / len(losses) if losses else 0,
            usage=Usage(train_tokens=total_tokens),
        )

    async def probe(self, sampler_checkpoint: str) -> bool:
        service = self._service()
        sampling = await service.create_sampling_client_async(model_path=sampler_checkpoint)
        tokenizer = sampling.get_tokenizer()
        from tinker import types

        result = await sampling.sample_async(
            prompt=types.ModelInput.from_ints(tokenizer.encode("User: 2+2?\nAssistant:")),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=8, temperature=0),
        )
        return bool(result.sequences and result.sequences[0].tokens)

    async def evaluate(
        self, sampler_checkpoint: str, spec: EvolutionSpec, genotype: Genotype
    ) -> EvaluationResult:
        supported_benchmark = "arithmetic_exact_match"
        unsupported = [item.name for item in spec.benchmarks if item.name != supported_benchmark]
        if unsupported:
            raise ValueError(
                "The live Tinker adapter does not have evaluators for: " + ", ".join(unsupported)
            )

        from tinker import types

        service = self._service(spec)
        sampling = await service.create_sampling_client_async(model_path=sampler_checkpoint)
        tokenizer = sampling.get_tokenizer()
        tasks = [
            ("Calculate 19 * 17. Give only the answer.", "323"),
            ("What is 225 divided by 15? Give only the answer.", "15"),
            ("A team places 18 beams per day for 6 days. Total?", "108"),
            ("What is 31 + 47? Give only the answer.", "78"),
        ]
        requested_examples = spec.benchmarks[0].examples
        if requested_examples > len(tasks):
            raise ValueError(
                f"{supported_benchmark} supports at most {len(tasks)} examples in v0.1"
            )
        tasks = tasks[:requested_examples]
        correct = 0
        prompt_tokens = 0
        sample_tokens = 0
        examples = []
        started = time.perf_counter()
        for prompt, answer in tasks:
            tokens = tokenizer.encode(f"User: {prompt}\nAssistant:")
            prompt_tokens += len(tokens)
            result = await sampling.sample_async(
                prompt=types.ModelInput.from_ints(tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=64, temperature=genotype.temperature
                ),
            )
            sequence = result.sequences[0]
            sample_tokens += len(sequence.tokens)
            text = tokenizer.decode(sequence.tokens)
            matched = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            reward = int(bool(matched and matched.group() == answer))
            correct += reward
            examples.append({"prompt": prompt, "response": text, "reward": reward})
        accuracy = correct / len(tasks)
        scores = {supported_benchmark: accuracy}
        return EvaluationResult(
            scores=scores,
            latency_ms=(time.perf_counter() - started) * 1000 / len(tasks),
            usage=Usage(prompt_tokens=prompt_tokens, sample_tokens=sample_tokens),
            examples=examples,
        )

    async def chat(
        self, sampler_checkpoint: str, messages: list[dict[str, str]], **parameters: object
    ) -> dict[str, object]:
        base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
        payload = {"model": sampler_checkpoint, "messages": messages, **parameters}
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._require_key()}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()

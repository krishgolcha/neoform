from __future__ import annotations

import asyncio
import math
import random
from typing import Any

from .adapters.base import LabAdapter
from .budget import BudgetExceeded, BudgetLedger
from .domain import (
    CandidateStatus,
    Event,
    EvolutionSpec,
    EvolutionStatus,
    Genotype,
)
from .pricing import ModelPrice, PricingCatalog
from .storage import Store


class EvolutionEngine:
    def __init__(self, store: Store, adapter: LabAdapter, pricing: PricingCatalog):
        self.store = store
        self.adapter = adapter
        self.pricing = pricing
        self._controls: dict[str, asyncio.Event] = {}

    def create(self, spec: EvolutionSpec) -> str:
        estimate = self.pricing.estimate(spec)
        if estimate.reserved_usd > spec.max_usd:
            raise BudgetExceeded(
                f"Worst-case estimate ${estimate.reserved_usd:.2f} exceeds cap ${spec.max_usd:.2f}"
            )
        return self.store.create_evolution(spec)

    def _genotype(self, spec: EvolutionSpec, generation: int, ordinal: int) -> Genotype:
        rng = random.Random(f"{spec.seed}:{generation}:{ordinal}")
        low = math.log(spec.mutations.learning_rate_min)
        high = math.log(spec.mutations.learning_rate_max)
        return Genotype(
            learning_rate=math.exp(rng.uniform(low, high)),
            temperature=rng.choice(spec.mutations.temperatures),
            loss_function=rng.choice(spec.mutations.loss_functions),
            advantage_clip=rng.choice(spec.mutations.advantage_clips),
            optimizer_mode=rng.choice(spec.mutations.optimizer_modes),
            curriculum_seed=rng.randint(1, 2**31 - 1),
        )

    def pause(self, evolution_id: str) -> None:
        self.store.set_evolution_status(evolution_id, EvolutionStatus.PAUSED)
        self._controls.setdefault(evolution_id, asyncio.Event()).clear()

    def resume(self, evolution_id: str) -> None:
        self._controls.setdefault(evolution_id, asyncio.Event()).set()
        self.store.set_evolution_status(evolution_id, EvolutionStatus.RUNNING)

    def cancel(self, evolution_id: str) -> None:
        self.store.set_evolution_status(evolution_id, EvolutionStatus.CANCELLED)
        self._controls.setdefault(evolution_id, asyncio.Event()).set()

    async def run(self, evolution_id: str) -> None:
        evolution = self.store.get_evolution(evolution_id)
        if not evolution:
            raise KeyError(evolution_id)
        spec = EvolutionSpec.model_validate(evolution["spec"])
        price = self.pricing.get(spec.base_model)
        estimate = self.pricing.estimate(spec)
        ledger = BudgetLedger(spec.max_usd, evolution["spent"], evolution["reserved"])
        control = self._controls.setdefault(evolution_id, asyncio.Event())
        control.set()
        self.store.set_evolution_status(evolution_id, EvolutionStatus.RUNNING)
        try:
            survivors: list[dict[str, Any]] = []
            candidate_reservation = estimate.reserved_usd / (
                spec.search.population * spec.search.generations
            )
            for generation in range(spec.search.generations):
                await self._wait_if_paused(evolution_id, control)
                parents = survivors or [None]
                ids = []
                for ordinal in range(spec.search.population):
                    parent = parents[ordinal % len(parents)]
                    candidate_id = self.store.create_candidate(
                        evolution_id,
                        generation,
                        ordinal,
                        self._genotype(spec, generation, ordinal),
                        parent["id"] if parent else None,
                    )
                    ids.append(candidate_id)
                semaphore = asyncio.Semaphore(spec.search.max_concurrency)

                async def execute(
                    candidate_id: str, candidate_semaphore: asyncio.Semaphore = semaphore
                ) -> None:
                    async with candidate_semaphore:
                        await self._run_candidate(
                            evolution_id,
                            candidate_id,
                            spec,
                            price,
                            ledger,
                            candidate_reservation,
                            control,
                        )

                await asyncio.gather(*(execute(candidate_id) for candidate_id in ids))
                completed = [
                    item
                    for item in self.store.list_candidates(evolution_id)
                    if item["generation"] == generation
                    and item["status"] == CandidateStatus.COMPLETE
                ]
                if not completed:
                    raise RuntimeError(f"Generation {generation} produced no valid candidates")
                survivors = sorted(
                    completed,
                    key=lambda item: (-float(item["score"]), item["id"]),
                )[: spec.search.survivors]
                self.store.add_event(
                    Event(
                        evolution_id=evolution_id,
                        type="generation.completed",
                        data={
                            "generation": generation,
                            "survivors": [item["id"] for item in survivors],
                        },
                    )
                )
            self.store.set_evolution_status(evolution_id, EvolutionStatus.COMPLETED)
        except Exception as exc:
            current = self.store.get_evolution(evolution_id)
            if current and current["status"] != EvolutionStatus.CANCELLED:
                self.store.set_evolution_status(evolution_id, EvolutionStatus.FAILED)
                self.store.add_event(
                    Event(
                        evolution_id=evolution_id,
                        type="evolution.error",
                        data={"message": str(exc)},
                    )
                )
            raise

    async def _wait_if_paused(self, evolution_id: str, control: asyncio.Event) -> None:
        while True:
            evolution = self.store.get_evolution(evolution_id)
            if not evolution or evolution["status"] == EvolutionStatus.CANCELLED:
                raise asyncio.CancelledError
            if evolution["status"] != EvolutionStatus.PAUSED:
                return
            await control.wait()

    async def _run_candidate(
        self,
        evolution_id: str,
        candidate_id: str,
        spec: EvolutionSpec,
        price: ModelPrice,
        ledger: BudgetLedger,
        reserve_amount: float,
        control: asyncio.Event,
    ) -> None:
        reservation = await ledger.reserve(candidate_id, reserve_amount)
        self.store.set_budget(evolution_id, ledger.spent, ledger.reserved)
        candidates = self.store.list_candidates(evolution_id)
        candidate = next(item for item in candidates if item["id"] == candidate_id)
        parent = next(
            (item for item in candidates if item["id"] == candidate["parent_id"]), None
        )
        genotype = Genotype.model_validate(candidate["genotype"])
        try:
            await self._wait_if_paused(evolution_id, control)
            self.store.update_candidate(candidate_id, status=CandidateStatus.TRAINING)
            training = await self.adapter.train(
                candidate_id,
                spec,
                genotype,
                parent["state_checkpoint"] if parent else None,
            )
            self.store.update_candidate(
                candidate_id,
                status=CandidateStatus.PROBING,
                state_checkpoint=training.state_checkpoint,
                sampler_checkpoint=training.sampler_checkpoint,
            )
            if not await self.adapter.probe(training.sampler_checkpoint):
                self.store.update_candidate(
                    candidate_id, status=CandidateStatus.INVALID, error="Checkpoint probe failed"
                )
                await ledger.reconcile(reservation, price.cost(training.usage))
                return
            self.store.update_candidate(candidate_id, status=CandidateStatus.EVALUATING)
            evaluation = await self.adapter.evaluate(training.sampler_checkpoint, spec, genotype)
            score = sum(
                benchmark.weight * evaluation.scores.get(benchmark.name, 0)
                for benchmark in spec.benchmarks
            )
            metrics = {
                "scores": evaluation.scores,
                "training_loss": training.training_loss,
                "latency_ms": evaluation.latency_ms,
                "usage": (training.usage + evaluation.usage).model_dump()
                if hasattr(training.usage, "__add__")
                else {
                    "train_tokens": training.usage.train_tokens,
                    "sample_tokens": evaluation.usage.sample_tokens,
                    "prompt_tokens": evaluation.usage.prompt_tokens,
                },
                "examples": evaluation.examples,
            }
            actual_usage = training.usage.model_copy(
                update={
                    "sample_tokens": evaluation.usage.sample_tokens,
                    "prompt_tokens": evaluation.usage.prompt_tokens,
                }
            )
            await ledger.reconcile(reservation, price.cost(actual_usage))
            self.store.update_candidate(
                candidate_id,
                status=CandidateStatus.COMPLETE,
                score=score,
                metrics=metrics,
            )
            self.store.add_event(
                Event(
                    evolution_id=evolution_id,
                    type="candidate.completed",
                    data={"candidate_id": candidate_id, "score": score},
                )
            )
        except asyncio.CancelledError:
            await ledger.release(reservation)
            raise
        except Exception as exc:
            await ledger.release(reservation)
            self.store.update_candidate(
                candidate_id, status=CandidateStatus.FAILED, error=str(exc)
            )
            self.store.add_event(
                Event(
                    evolution_id=evolution_id,
                    type="candidate.failed",
                    data={"candidate_id": candidate_id, "message": str(exc)},
                )
            )
        finally:
            self.store.set_budget(evolution_id, ledger.spent, ledger.reserved)

from __future__ import annotations

from abc import ABC, abstractmethod

from neoform.domain import EvaluationResult, EvolutionSpec, Genotype, TrainingResult


class LabAdapter(ABC):
    @abstractmethod
    async def doctor(self) -> dict[str, object]: ...

    @abstractmethod
    async def train(
        self,
        candidate_id: str,
        spec: EvolutionSpec,
        genotype: Genotype,
        parent_state: str | None,
    ) -> TrainingResult: ...

    @abstractmethod
    async def probe(self, sampler_checkpoint: str) -> bool: ...

    @abstractmethod
    async def evaluate(
        self, sampler_checkpoint: str, spec: EvolutionSpec, genotype: Genotype
    ) -> EvaluationResult: ...

    @abstractmethod
    async def chat(
        self, sampler_checkpoint: str, messages: list[dict[str, str]], **parameters: object
    ) -> dict[str, object]: ...


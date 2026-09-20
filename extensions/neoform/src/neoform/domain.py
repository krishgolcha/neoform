from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class EvolutionStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CandidateStatus(StrEnum):
    QUEUED = "queued"
    TRAINING = "training"
    PROBING = "probing"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    INVALID = "invalid"
    FAILED = "failed"


class BenchmarkSpec(BaseModel):
    name: str
    weight: Annotated[float, Field(gt=0, le=1)]
    examples: Annotated[int, Field(ge=1, le=10_000)] = 50

    @model_validator(mode="after")
    def validate_registered_evaluator(self) -> BenchmarkSpec:
        from neoform.evaluators import available_names

        known = available_names()
        if self.name not in known:
            raise ValueError(
                f"Unknown evaluator {self.name!r}. Available: {', '.join(known)}"
            )
        return self


class SearchSpec(BaseModel):
    population: Annotated[int, Field(ge=2, le=32)] = 4
    generations: Annotated[int, Field(ge=1, le=20)] = 3
    survivors: Annotated[int, Field(ge=1, le=16)] = 2
    rank: Annotated[int, Field(ge=1, le=256)] = 32
    steps_per_candidate: Annotated[int, Field(ge=1, le=10_000)] = 25
    batch_size: Annotated[int, Field(ge=1, le=128)] = 8
    max_sequence_tokens: Annotated[int, Field(ge=64, le=32_768)] = 1024
    max_concurrency: Annotated[int, Field(ge=1, le=16)] = 2

    @model_validator(mode="after")
    def validate_survivors(self) -> SearchSpec:
        if self.survivors >= self.population:
            raise ValueError("survivors must be smaller than population")
        return self


class MutationSpace(BaseModel):
    learning_rate_min: float = Field(default=2e-5, gt=0)
    learning_rate_max: float = Field(default=2e-4, gt=0)
    temperatures: list[float] = Field(default_factory=lambda: [0.0, 0.6, 0.8, 1.0])
    max_tokens: list[int] = Field(default_factory=lambda: [32, 64, 128, 256])
    top_p: list[float] = Field(default_factory=lambda: [0.9, 1.0])
    loss_functions: list[str] = Field(default_factory=lambda: ["cross_entropy"])
    advantage_clips: list[float] = Field(default_factory=lambda: [1.0])
    optimizer_modes: list[str] = Field(default_factory=lambda: ["resume", "reset"])
    curriculum_seed_min: int = Field(default=1, ge=1)
    curriculum_seed_max: int = Field(default=2**31 - 1, ge=1)
    parent_selection: Literal["rank", "tournament", "softmax"] = "rank"
    tournament_size: Annotated[int, Field(ge=2, le=16)] = 2
    softmax_temperature: float = Field(default=0.15, gt=0)
    mutation_rate: float = Field(default=0.35, ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> MutationSpace:
        if self.learning_rate_min >= self.learning_rate_max:
            raise ValueError("learning_rate_min must be below learning_rate_max")
        if not self.temperatures or not all(0 <= value <= 2 for value in self.temperatures):
            raise ValueError("temperatures must be between 0 and 2")
        if not self.max_tokens or not all(8 <= value <= 8192 for value in self.max_tokens):
            raise ValueError("max_tokens values must be between 8 and 8192")
        if not self.top_p or not all(0 < value <= 1 for value in self.top_p):
            raise ValueError("top_p values must be in (0, 1]")
        if not self.loss_functions:
            raise ValueError("loss_functions must not be empty")
        if not self.optimizer_modes or any(
            mode not in {"resume", "reset"} for mode in self.optimizer_modes
        ):
            raise ValueError("optimizer_modes must be resume and/or reset")
        if self.curriculum_seed_min >= self.curriculum_seed_max:
            raise ValueError("curriculum_seed_min must be below curriculum_seed_max")
        return self


class EvolutionSpec(BaseModel):
    name: str = "Reasoning frontier"
    project_id: str | None = None
    base_model: str = "Qwen/Qwen3.5-4B"
    seed: int = 2026
    max_usd: Annotated[float, Field(gt=0, le=100_000)]
    checkpoint_ttl_days: Annotated[int, Field(ge=1, le=365)] = 7
    search: SearchSpec = Field(default_factory=SearchSpec)
    mutations: MutationSpace = Field(default_factory=MutationSpace)
    benchmarks: list[BenchmarkSpec] = Field(
        default_factory=lambda: [
            BenchmarkSpec(name="arithmetic_exact_match", weight=1.0, examples=4),
        ]
    )

    @model_validator(mode="after")
    def validate_benchmarks(self) -> EvolutionSpec:
        if abs(sum(item.weight for item in self.benchmarks) - 1.0) > 1e-6:
            raise ValueError("benchmark weights must sum to 1")
        return self


class Genotype(BaseModel):
    learning_rate: float
    temperature: float
    max_tokens: int = 64
    top_p: float = 1.0
    loss_function: str
    advantage_clip: float
    optimizer_mode: str
    curriculum_seed: int


class Usage(BaseModel):
    train_tokens: int = 0
    sample_tokens: int = 0
    prompt_tokens: int = 0


class TrainingResult(BaseModel):
    state_checkpoint: str
    sampler_checkpoint: str
    training_loss: float
    usage: Usage


class EvaluationResult(BaseModel):
    scores: dict[str, float]
    latency_ms: float
    usage: Usage
    examples: list[dict[str, Any]] = Field(default_factory=list)


class CostEstimate(BaseModel):
    train_usd: float
    sample_usd: float
    reserved_usd: float
    pricing_source: str


class Event(BaseModel):
    sequence: int | None = None
    evolution_id: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

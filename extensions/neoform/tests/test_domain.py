import pytest
from pydantic import ValidationError

from neoform.domain import BenchmarkSpec, EvolutionSpec, MutationSpace, SearchSpec
from neoform.evaluators import available_names


def test_evolution_requires_normalized_benchmark_weights():
    with pytest.raises(ValidationError, match="weights must sum to 1"):
        EvolutionSpec(
            max_usd=5,
            benchmarks=[BenchmarkSpec(name="arithmetic_exact_match", weight=0.4)],
        )


def test_survivor_count_must_be_smaller_than_population():
    with pytest.raises(ValidationError, match="survivors must be smaller"):
        SearchSpec(population=2, survivors=2)


def test_unknown_evaluator_is_rejected():
    with pytest.raises(ValidationError, match="Unknown evaluator"):
        BenchmarkSpec(name="unimplemented", weight=1, examples=1)


def test_default_benchmark_is_registered():
    spec = EvolutionSpec(max_usd=5)
    assert spec.benchmarks[0].name in available_names()


def test_mutation_space_rejects_inverted_learning_rates():
    with pytest.raises(ValidationError, match="learning_rate_min"):
        MutationSpace(learning_rate_min=1e-3, learning_rate_max=1e-4)

import pytest
from pydantic import ValidationError

from neoform.domain import BenchmarkSpec, EvolutionSpec, SearchSpec


def test_evolution_requires_normalized_benchmark_weights():
    with pytest.raises(ValidationError, match="weights must sum to 1"):
        EvolutionSpec(
            max_usd=5,
            benchmarks=[BenchmarkSpec(name="one", weight=0.4)],
        )


def test_survivor_count_must_be_smaller_than_population():
    with pytest.raises(ValidationError, match="survivors must be smaller"):
        SearchSpec(population=2, survivors=2)

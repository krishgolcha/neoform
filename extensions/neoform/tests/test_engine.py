from pathlib import Path

import pytest

from neoform.adapters.mock import MockTinkerAdapter
from neoform.domain import BenchmarkSpec, EvolutionSpec, SearchSpec
from neoform.engine import EvolutionEngine
from neoform.pricing import ModelPrice, PricingCatalog
from neoform.storage import Store


def catalog():
    return PricingCatalog(
        {"Qwen/Qwen3.5-4B": ModelPrice("Qwen/Qwen3.5-4B", 0.737, 1.005, 0.33)},
        "test",
    )


def spec():
    return EvolutionSpec(
        name="test evolution",
        max_usd=5,
        search=SearchSpec(
            population=2,
            generations=2,
            survivors=1,
            steps_per_candidate=1,
            batch_size=1,
            max_sequence_tokens=64,
        ),
        benchmarks=[BenchmarkSpec(name="gsm8k", weight=1, examples=2)],
    )


@pytest.mark.asyncio
async def test_engine_runs_generations_and_promotes_manually(tmp_path: Path):
    store = Store(tmp_path / "lab.db")
    engine = EvolutionEngine(store, MockTinkerAdapter(), catalog())
    evolution_id = engine.create(spec())
    await engine.run(evolution_id)

    evolution = store.get_evolution(evolution_id)
    assert evolution is not None
    assert evolution["status"] == "completed"
    assert evolution["champion_id"] is None
    assert len(evolution["candidates"]) == 4
    assert all(item["status"] == "complete" for item in evolution["candidates"])
    assert evolution["spent"] > 0

    champion = max(evolution["candidates"], key=lambda item: item["score"])
    store.promote(evolution_id, champion["id"])
    assert store.get_evolution(evolution_id)["champion_id"] == champion["id"]


@pytest.mark.asyncio
async def test_resume_does_not_duplicate_completed_candidates(tmp_path: Path):
    store = Store(tmp_path / "lab.db")
    engine = EvolutionEngine(store, MockTinkerAdapter(), catalog())
    evolution_id = engine.create(spec())
    await engine.run(evolution_id)
    await engine.run(evolution_id)
    assert len(store.list_candidates(evolution_id)) == 4

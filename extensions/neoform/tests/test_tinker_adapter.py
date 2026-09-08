import pytest

from neoform.adapters.tinker import TinkerAdapter
from neoform.domain import BenchmarkSpec, EvolutionSpec, Genotype


def genotype(loss_function: str = "cross_entropy") -> Genotype:
    return Genotype(
        learning_rate=1e-4,
        temperature=0.6,
        loss_function=loss_function,
        advantage_clip=1.0,
        optimizer_mode="reset",
        curriculum_seed=2026,
    )


@pytest.mark.asyncio
async def test_live_adapter_rejects_an_unimplemented_loss_before_spending():
    adapter = TinkerAdapter(api_key="test-key")

    with pytest.raises(ValueError, match="only cross_entropy"):
        await adapter.train("candidate", EvolutionSpec(max_usd=5), genotype("ppo"), None)


@pytest.mark.asyncio
async def test_live_adapter_rejects_an_unknown_evaluator_before_spending():
    adapter = TinkerAdapter(api_key="test-key")
    spec = EvolutionSpec(
        max_usd=5,
        benchmarks=[BenchmarkSpec(name="unimplemented", weight=1, examples=1)],
    )

    with pytest.raises(ValueError, match="does not have evaluators for: unimplemented"):
        await adapter.evaluate("tinker://test/sampler", spec, genotype())

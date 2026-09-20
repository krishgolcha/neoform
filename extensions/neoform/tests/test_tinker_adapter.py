import pytest

from neoform.adapters.tinker import TinkerAdapter
from neoform.domain import EvolutionSpec, Genotype
from neoform.mutations import reject_unsupported_loss


def genotype(loss_function: str = "cross_entropy") -> Genotype:
    return Genotype(
        learning_rate=1e-4,
        temperature=0.6,
        max_tokens=64,
        top_p=1.0,
        loss_function=loss_function,
        advantage_clip=1.0,
        optimizer_mode="reset",
        curriculum_seed=2026,
    )


async def test_live_adapter_rejects_an_unimplemented_loss_before_spending():
    adapter = TinkerAdapter(api_key="test-key")

    with pytest.raises(ValueError, match="only cross_entropy"):
        await adapter.train("candidate", EvolutionSpec(max_usd=5), genotype("ppo"), None)


def test_unsupported_loss_helper_is_honest():
    with pytest.raises(ValueError, match="Unsupported loss function 'cispo'"):
        reject_unsupported_loss("cispo")

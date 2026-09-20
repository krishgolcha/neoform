from pathlib import Path

from neoform.domain import EvolutionSpec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_manifests_validate():
    files = sorted(EXAMPLES.glob("*.json"))
    assert {path.name for path in files} >= {
        "smoke-arithmetic.json",
        "reasoning-gsm8k.json",
        "instruction-following.json",
        "reasoning-frontier.json",
    }
    for path in files:
        spec = EvolutionSpec.model_validate_json(path.read_text())
        assert spec.benchmarks
        assert spec.max_usd > 0

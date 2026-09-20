import pytest

from neoform.domain import BenchmarkSpec, EvolutionSpec, MutationSpace
from neoform.evaluators import get_evaluator
from neoform.evaluators.numbers import extract_final_number
from neoform.mutations import sample_genotype, select_parent


def test_arithmetic_dataset_is_not_hardcoded_tuples():
    evaluator = get_evaluator("arithmetic_exact_match")
    items = evaluator.load_items(
        BenchmarkSpec(name="arithmetic_exact_match", weight=1, examples=4), seed=7
    )
    assert len(items) == 4
    assert evaluator.score(items[0], items[0].gold) == 1.0
    assert evaluator.score(items[0], "wrong") == 0.0
    assert evaluator.catalog_size() >= 16


def test_gsm8k_extracts_final_number():
    evaluator = get_evaluator("gsm8k_exact_match")
    item = evaluator.all_items()[0]
    assert extract_final_number("scratch 10 then #### 72") == "72"
    assert evaluator.score(item, "The boxes total 72.") == 1.0
    assert evaluator.score(item, "The boxes total 71.") == 0.0
    loaded = evaluator.load_items(
        BenchmarkSpec(name="gsm8k_exact_match", weight=1, examples=8), seed=1
    )
    assert len(loaded) == 8


def test_instruction_following_checks_json_and_format():
    evaluator = get_evaluator("instruction_following")
    items = {item.id: item for item in evaluator.all_items()}
    json_item = items["if-001"]
    assert evaluator.score(json_item, '{"status": "ok", "count": 3}') == 1.0
    assert evaluator.score(json_item, '{"status": "ok"}') == 0.0
    bullets = items["if-002"]
    assert evaluator.score(bullets, "- apple\n- banana\n- cherry") == 1.0
    assert evaluator.score(bullets, "- apple\n- banana") == 0.0
    yes = items["if-003"]
    assert evaluator.score(yes, "YES") == 1.0
    assert evaluator.score(yes, "yes") == 0.0
    numbered = items["if-006"]
    assert evaluator.score(numbered, "1. Collect checkpoints\n2. Measure fitness") == 1.0
    contains = items["if-008"]
    assert evaluator.score(contains, "Train LoRA candidates on Tinker safely.") == 1.0
    fenced = items["if-001"]
    assert (
        evaluator.score(
            fenced,
            "```json\n{\"status\": \"ok\", \"count\": 3}\n```",
        )
        == 1.0
    )


def test_unknown_evaluator_name_raises():
    with pytest.raises(ValueError, match="Unknown evaluator"):
        get_evaluator("not_a_real_task")


def test_genotype_sampling_is_seeded():
    spec = EvolutionSpec(max_usd=5, seed=99)
    first = sample_genotype(spec, 0, 1)
    second = sample_genotype(spec, 0, 1)
    third = sample_genotype(spec, 0, 2)
    assert first == second
    assert first != third
    assert spec.mutations.learning_rate_min <= first.learning_rate
    assert first.learning_rate <= spec.mutations.learning_rate_max
    assert first.max_tokens in spec.mutations.max_tokens
    assert first.top_p in spec.mutations.top_p


def test_parent_mutation_stays_inside_the_space():
    spec = EvolutionSpec(max_usd=5, seed=3)
    parent = sample_genotype(spec, 0, 0)
    child = sample_genotype(spec, 1, 0, parent)
    assert spec.mutations.learning_rate_min <= child.learning_rate
    assert child.learning_rate <= spec.mutations.learning_rate_max
    assert child.max_tokens in spec.mutations.max_tokens


def test_tournament_and_softmax_parent_selection_are_seeded():
    survivors = [
        {"id": "a", "score": 0.2},
        {"id": "b", "score": 0.9},
        {"id": "c", "score": 0.4},
    ]
    tournament = EvolutionSpec(
        max_usd=5,
        seed=11,
        mutations=MutationSpace(parent_selection="tournament", tournament_size=2),
    )
    softmax = EvolutionSpec(
        max_usd=5,
        seed=11,
        mutations=MutationSpace(parent_selection="softmax", softmax_temperature=0.2),
    )
    assert select_parent(survivors, tournament, 1, 0) == select_parent(
        survivors, tournament, 1, 0
    )
    assert select_parent(survivors, softmax, 2, 1) == select_parent(survivors, softmax, 2, 1)
    assert select_parent([], tournament, 0, 0) is None

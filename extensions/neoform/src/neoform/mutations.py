from __future__ import annotations

import math
import random

from .domain import EvolutionSpec, Genotype

LIVE_LOSS_FUNCTIONS = frozenset({"cross_entropy"})


def reject_unsupported_loss(loss_function: str) -> None:
    if loss_function not in LIVE_LOSS_FUNCTIONS:
        supported = ", ".join(sorted(LIVE_LOSS_FUNCTIONS))
        raise ValueError(
            f"Unsupported loss function {loss_function!r}. "
            f"The live Tinker adapter currently supports only {supported}."
        )


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _maybe_choice(rng: random.Random, choices: list, current, mutation_rate: float):
    if current in choices and rng.random() > mutation_rate:
        return current
    return rng.choice(choices)


def sample_genotype(
    spec: EvolutionSpec,
    generation: int,
    ordinal: int,
    parent: Genotype | None = None,
) -> Genotype:
    rng = random.Random(f"{spec.seed}:geno:{generation}:{ordinal}")
    space = spec.mutations
    if parent is None:
        return Genotype(
            learning_rate=_log_uniform(rng, space.learning_rate_min, space.learning_rate_max),
            temperature=rng.choice(space.temperatures),
            max_tokens=rng.choice(space.max_tokens),
            top_p=rng.choice(space.top_p),
            loss_function=rng.choice(space.loss_functions),
            advantage_clip=rng.choice(space.advantage_clips),
            optimizer_mode=rng.choice(space.optimizer_modes),
            curriculum_seed=rng.randint(space.curriculum_seed_min, space.curriculum_seed_max),
        )

    span = (math.log(space.learning_rate_max) - math.log(space.learning_rate_min)) * 0.25
    mutated_lr = math.exp(rng.gauss(math.log(parent.learning_rate), span))
    mutated_lr = min(space.learning_rate_max, max(space.learning_rate_min, mutated_lr))
    if rng.random() > space.mutation_rate and (
        space.curriculum_seed_min <= parent.curriculum_seed <= space.curriculum_seed_max
    ):
        curriculum_seed = parent.curriculum_seed
    else:
        curriculum_seed = rng.randint(space.curriculum_seed_min, space.curriculum_seed_max)
    return Genotype(
        learning_rate=mutated_lr,
        temperature=_maybe_choice(rng, space.temperatures, parent.temperature, space.mutation_rate),
        max_tokens=_maybe_choice(rng, space.max_tokens, parent.max_tokens, space.mutation_rate),
        top_p=_maybe_choice(rng, space.top_p, parent.top_p, space.mutation_rate),
        loss_function=_maybe_choice(
            rng, space.loss_functions, parent.loss_function, space.mutation_rate
        ),
        advantage_clip=_maybe_choice(
            rng, space.advantage_clips, parent.advantage_clip, space.mutation_rate
        ),
        optimizer_mode=_maybe_choice(
            rng, space.optimizer_modes, parent.optimizer_mode, space.mutation_rate
        ),
        curriculum_seed=curriculum_seed,
    )


def select_parent(
    survivors: list[dict],
    spec: EvolutionSpec,
    generation: int,
    ordinal: int,
) -> dict | None:
    if not survivors:
        return None
    rng = random.Random(f"{spec.seed}:parent:{generation}:{ordinal}")
    mode = spec.mutations.parent_selection
    if mode == "rank":
        return survivors[ordinal % len(survivors)]
    if mode == "tournament":
        size = min(spec.mutations.tournament_size, len(survivors))
        pool = [survivors[rng.randrange(len(survivors))] for _ in range(size)]
        return max(pool, key=lambda item: (float(item["score"]), item["id"]))
    scores = [float(item["score"]) for item in survivors]
    peak = max(scores)
    temperature = spec.mutations.softmax_temperature
    weights = [math.exp((score - peak) / temperature) for score in scores]
    pick = rng.random() * sum(weights)
    cumulative = 0.0
    for item, weight in zip(survivors, weights, strict=True):
        cumulative += weight
        if pick <= cumulative:
            return item
    return survivors[-1]

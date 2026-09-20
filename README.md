<div align="center">
  <h1>NEOFORM</h1>
  <p><strong>Open-source evolutionary post-training for Tinker.</strong></p>
  <p>
    Branch LoRA candidates. Measure real fitness. Promote only the model that earns it.
  </p>
  <p>
    <a href="https://github.com/krishgolcha/neoform/actions">CI</a> ·
    <a href="https://www.apache.org/licenses/LICENSE-2.0">Apache-2.0</a> ·
    <a href="https://github.com/krishgolcha/neoform/releases">Releases</a> ·
    <a href="https://tinker-docs.thinkingmachines.ai/">Tinker documentation</a>
  </p>
</div>

![NEOFORM evolution lab](docs/images/neoform-lab.png)

NEOFORM is a local lab above the [Tinker API](https://github.com/thinking-machines-lab/tinker).
It spawns competing LoRA candidates, mutates live-applicable hyperparameters, evaluates
checkpoints on pluggable benchmarks, enforces a USD spend cap, and leaves champion
promotion to you.

> [!IMPORTANT]
> NEOFORM is an independent, unofficial open-source project created and maintained
> by **Krish Golcha**. It is not endorsed by or affiliated with Thinking Machines Lab.

## Architecture

```text
Next.js lab ── REST + SSE ── FastAPI runtime ── Evolution engine
                                      │              │
                                      │              ├─ Budget ledger (worst-case USD cap)
                                      │              ├─ Seeded mutations + parent selection
                                      │              ├─ Pluggable evaluators (weighted)
                                      │              └─ SQLite event store
                                      │
                                      └─ Tinker SDK ── train / save / sample / probe
```

NEOFORM creates `WEIGHTS` checkpoints for branching and `SAMPLER_WEIGHTS`
checkpoints for evaluation. It does not merge or cross over LoRA weights.
Unsupported loss functions are rejected instead of being silently treated as PPO.

## Quick start

Requirements: Python 3.11+, `uv`, Node.js 20+, `pnpm`, a funded Tinker account,
and a Tinker API key.

```bash
git clone https://github.com/krishgolcha/neoform.git
cd neoform
git checkout main
export TINKER_API_KEY="your-key"

cd extensions/neoform
uv sync --extra dev
uv run neoform doctor
uv run neoform evaluators
uv run neoform serve
```

In a second terminal:

```bash
cd extensions/neoform/web
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). NEOFORM will not offer a
simulated product mode when Tinker is unavailable; the mock adapter is reserved for tests.

## Start from a manifest

Cheap smoke (tiny search, arithmetic dataset):

```bash
cd extensions/neoform
uv run neoform estimate examples/smoke-arithmetic.json
uv run neoform evolve examples/smoke-arithmetic.json
uv run neoform status
```

Other bundled manifests:

| File | Evaluator | Notes |
| --- | --- | --- |
| `examples/smoke-arithmetic.json` | `arithmetic_exact_match` | Smallest live smoke |
| `examples/reasoning-gsm8k.json` | `gsm8k_exact_match` | Compact GSM8K-style items, no Hugging Face download |
| `examples/instruction-following.json` | `instruction_following` | JSON / format / constraint checks |
| `examples/reasoning-frontier.json` | weighted mix | All three evaluators |

CLI: `neoform doctor`, `neoform evaluators`, `neoform estimate`, `neoform serve`,
`neoform evolve`, `neoform status`, `neoform promote`, `neoform export`.

## Evaluators

Registry keyed by `BenchmarkSpec.name`. Weighted aggregation is unchanged: the
candidate fitness is `sum(weight * score[name])`.

| Name | Scoring |
| --- | --- |
| `arithmetic_exact_match` | Dataset file of arithmetic items; extract-and-match the first number |
| `gsm8k_exact_match` | Bundled compact GSM8K-style items; extract the final number (`####` preferred) |
| `instruction_following` | JSON parse, required keys, regex, lists, prefix/word constraints |

Register another evaluator by implementing `Evaluator` and adding it to
`neoform.evaluators.registry`. The live Tinker adapter trains and evaluates
whatever benchmarks you select.

## Mutations

`MutationSpace` samples a deterministic genotype from the evolution seed:

- log-uniform learning rate range
- temperature, max tokens, top-p
- curriculum seed range
- optimizer resume vs reset
- optional rank / tournament / softmax parent selection (still seeded)
- offspring mutation rate around a selected parent

The live adapter applies those knobs. Loss functions other than `cross_entropy`
are rejected before spend.

## API

OpenAPI lives at `http://127.0.0.1:8787/docs` while the engine is running.

- `GET /api/v1/evaluators`
- `POST /api/v1/evolutions/estimate`
- `POST /api/v1/evolutions`
- `POST /api/v1/evolutions/{id}/start|pause|resume|cancel`
- `GET /api/v1/evolutions/{id}/events`
- `POST /api/v1/evolutions/{id}/promote`
- `POST /v1/chat/completions`

Tinker's OpenAI-compatible inference remains a beta service intended for testing
and internal workflows. NEOFORM's champion gateway carries the same limitation.

## Development and tests

```bash
cd extensions/neoform
uv run ruff check src tests
uv run pytest --cov=neoform

cd web
pnpm lint
pnpm typecheck
pnpm build
pnpm test:e2e
```

The real API smoke test is opt-in and refuses to run without both
`TINKER_API_KEY` and `NEOFORM_LIVE_TEST=1`. Automated CI uses deterministic mocks
and does not spend Tinker credits.

## Upstream SDK

This repository is a fork of the official Apache-2.0-licensed Tinker Python SDK.
The original SDK remains at the repository root and its documentation is at
[tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai/).
Upstream attribution is preserved in [NOTICE](NOTICE).

## License

Apache License 2.0. Original NEOFORM components are copyright 2026 Krish Golcha.
Upstream Tinker components retain their original ownership and notices.

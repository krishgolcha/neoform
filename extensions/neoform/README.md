# NEOFORM engine

Durable orchestration, pluggable evaluators, Tinker adapter, API, and CLI.

```bash
uv sync --extra dev
uv run neoform evaluators
uv run neoform estimate examples/smoke-arithmetic.json
uv run neoform serve
```

Example manifests:

- `examples/smoke-arithmetic.json` — cheap live smoke
- `examples/reasoning-gsm8k.json` — bundled GSM8K-style reasoning
- `examples/instruction-following.json` — JSON and format constraints
- `examples/reasoning-frontier.json` — weighted mix of all three evaluators

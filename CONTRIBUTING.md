# Contributing to NEOFORM

Thank you for helping improve open-source post-training research.

## Development setup

1. Fork the repository and create a focused branch.
2. Install the engine with `cd extensions/neoform && uv sync --extra dev`.
3. Install the web lab with `cd web && pnpm install`.
4. Keep real credentials in environment variables. Never commit keys or traces
   containing private prompts.
5. Run the checks documented in the root README before opening a pull request.

## Contribution boundaries

- Keep changes to the upstream Tinker SDK separable from NEOFORM changes.
- Do not make live API calls in automated tests.
- Preserve deterministic seeds in evolution and evaluation tests.
- Any new spend-bearing operation must reserve a conservative maximum cost first.
- New destructive checkpoint operations require an explicit user confirmation path.

By submitting a contribution, you agree that it is licensed under Apache-2.0.

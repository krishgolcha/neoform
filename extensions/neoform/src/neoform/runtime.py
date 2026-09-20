from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from .adapters.base import LabAdapter
from .adapters.mock import MockTinkerAdapter
from .adapters.tinker import TinkerAdapter
from .domain import EvolutionSpec
from .engine import EvolutionEngine
from .pricing import ModelPrice, PricingCatalog
from .settings import Settings
from .storage import Store


class Runtime:
    def __init__(
        self,
        settings: Settings,
        store: Store | None = None,
        adapter: LabAdapter | None = None,
        pricing: PricingCatalog | None = None,
    ):
        self.settings = settings
        self.store = store or Store(settings.data_dir / "neoform.db")
        self.adapter = adapter or self._adapter(settings.adapter)
        self.pricing = pricing or PricingCatalog(
            {
                "Qwen/Qwen3.5-4B": ModelPrice(
                    "Qwen/Qwen3.5-4B", 0.737, 1.005, 0.33
                )
            },
            "bundled pricing fallback",
        )
        self.engine = EvolutionEngine(self.store, self.adapter, self.pricing)
        self.tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def _adapter(name: str) -> LabAdapter:
        if name == "mock":
            return MockTinkerAdapter()
        if name != "tinker":
            raise ValueError(f"Unknown NEOFORM_ADAPTER: {name}")
        return TinkerAdapter()

    async def refresh_pricing(self) -> PricingCatalog:
        self.pricing = await PricingCatalog.fetch(self.settings.data_dir / "models.json")
        self.engine.pricing = self.pricing
        return self.pricing

    async def estimate(self, spec: EvolutionSpec):
        if spec.base_model not in self.pricing.prices:
            await self.refresh_pricing()
        return self.pricing.estimate(spec)

    async def create(self, spec: EvolutionSpec) -> str:
        await self.estimate(spec)
        return self.engine.create(spec)

    def start(self, evolution_id: str) -> asyncio.Task[None]:
        current = self.tasks.get(evolution_id)
        if current and not current.done():
            return current
        task = asyncio.create_task(self.engine.run(evolution_id), name=f"neoform:{evolution_id}")
        self.tasks[evolution_id] = task
        return task

    async def shutdown(self) -> None:
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        for task in self.tasks.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def build_runtime(
    data_dir: Path | None = None,
    adapter: LabAdapter | None = None,
    pricing: PricingCatalog | None = None,
) -> Runtime:
    settings = Settings.from_env()
    if data_dir:
        settings = Settings(
            data_dir=data_dir,
            adapter=settings.adapter,
            host=settings.host,
            port=settings.port,
            web_origin=settings.web_origin,
        )
    return Runtime(settings, adapter=adapter, pricing=pricing)


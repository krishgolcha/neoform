from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .budget import BudgetExceeded
from .domain import EvolutionSpec
from .runtime import Runtime, build_runtime


class ChampionRequest(BaseModel):
    candidate_id: str


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    selected = runtime or build_runtime()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        selected.store.recover_interrupted()
        yield
        await selected.shutdown()

    application = FastAPI(
        title="NEOFORM API",
        description="Evolutionary post-training orchestration powered by Tinker",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.runtime = selected
    web_origins = {
        selected.settings.web_origin,
        selected.settings.web_origin.replace("localhost", "127.0.0.1"),
        selected.settings.web_origin.replace("127.0.0.1", "localhost"),
    }
    application.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(web_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    def get_evolution(evolution_id: str) -> dict[str, Any]:
        evolution = selected.store.get_evolution(evolution_id)
        if not evolution:
            raise HTTPException(404, "Evolution not found")
        return evolution

    @application.get("/api/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "service": "neoform", "version": "0.1.0"}

    @application.get("/api/v1/doctor")
    async def doctor() -> dict[str, object]:
        try:
            result = await selected.adapter.doctor()
            return {"ok": True, **result}
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    @application.post("/api/v1/evolutions/estimate")
    async def estimate(spec: EvolutionSpec):
        try:
            return await selected.estimate(spec)
        except (ValueError, BudgetExceeded) as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/v1/evolutions")
    async def list_evolutions():
        return {"data": selected.store.list_evolutions()}

    @application.post("/api/v1/evolutions", status_code=201)
    async def create_evolution(spec: EvolutionSpec):
        try:
            evolution_id = await selected.create(spec)
            return get_evolution(evolution_id)
        except (ValueError, BudgetExceeded) as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/v1/evolutions/{evolution_id}")
    async def evolution_detail(evolution_id: str):
        return get_evolution(evolution_id)

    @application.post("/api/v1/evolutions/{evolution_id}/start", status_code=202)
    async def start_evolution(evolution_id: str):
        evolution = get_evolution(evolution_id)
        if evolution["status"] not in {"draft", "paused", "failed"}:
            raise HTTPException(409, f"Cannot start evolution in {evolution['status']} state")
        selected.start(evolution_id)
        return {"id": evolution_id, "status": "running"}

    @application.post("/api/v1/evolutions/{evolution_id}/pause", status_code=202)
    async def pause_evolution(evolution_id: str):
        get_evolution(evolution_id)
        selected.engine.pause(evolution_id)
        return {"id": evolution_id, "status": "paused"}

    @application.post("/api/v1/evolutions/{evolution_id}/resume", status_code=202)
    async def resume_evolution(evolution_id: str):
        get_evolution(evolution_id)
        selected.engine.resume(evolution_id)
        selected.start(evolution_id)
        return {"id": evolution_id, "status": "running"}

    @application.post("/api/v1/evolutions/{evolution_id}/cancel", status_code=202)
    async def cancel_evolution(evolution_id: str):
        get_evolution(evolution_id)
        selected.engine.cancel(evolution_id)
        return {"id": evolution_id, "status": "cancelled"}

    @application.post("/api/v1/evolutions/{evolution_id}/promote")
    async def promote(evolution_id: str, body: ChampionRequest):
        get_evolution(evolution_id)
        try:
            selected.store.promote(evolution_id, body.candidate_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return get_evolution(evolution_id)

    @application.get("/api/v1/evolutions/{evolution_id}/events")
    async def events(
        request: Request,
        evolution_id: str,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ):
        get_evolution(evolution_id)
        after = int(last_event_id or 0)

        async def stream() -> AsyncIterator[str]:
            cursor = after
            while not await request.is_disconnected():
                rows = selected.store.events(evolution_id, cursor)
                for event in rows:
                    cursor = event["sequence"]
                    yield (
                        f"id: {cursor}\n"
                        f"event: {event['type']}\n"
                        f"data: {json.dumps(event)}\n\n"
                    )
                if not rows:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @application.post("/v1/chat/completions")
    async def champion_chat(body: ChatRequest, response: Response):
        prefix = "neoform://evolutions/"
        suffix = "/champion"
        if not body.model.startswith(prefix) or not body.model.endswith(suffix):
            raise HTTPException(422, "Model must be a NEOFORM champion alias")
        evolution_id = body.model[len(prefix) : -len(suffix)]
        evolution = get_evolution(evolution_id)
        if not evolution["champion_id"]:
            raise HTTPException(409, "This evolution has no promoted champion")
        candidate = next(
            item for item in evolution["candidates"] if item["id"] == evolution["champion_id"]
        )
        parameters = body.extra.copy()
        for key in ("temperature", "max_tokens", "reasoning_effort"):
            value = getattr(body, key)
            if value is not None:
                parameters[key] = value
        result = await selected.adapter.chat(
            candidate["sampler_checkpoint"], body.messages, **parameters
        )
        response.headers["X-Neoform-Evolution"] = evolution_id
        response.headers["X-Neoform-Candidate"] = candidate["id"]
        response.headers["X-Neoform-Checkpoint"] = candidate["sampler_checkpoint"]
        return result

    return application


app = create_app()

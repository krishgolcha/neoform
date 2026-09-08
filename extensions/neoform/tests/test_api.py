from pathlib import Path

from fastapi.testclient import TestClient

from neoform.adapters.mock import MockTinkerAdapter
from neoform.api import create_app
from neoform.pricing import ModelPrice, PricingCatalog
from neoform.runtime import build_runtime


def make_client(tmp_path: Path):
    pricing = PricingCatalog(
        {"Qwen/Qwen3.5-4B": ModelPrice("Qwen/Qwen3.5-4B", 0.737, 1.005, 0.33)},
        "test",
    )
    runtime = build_runtime(tmp_path, adapter=MockTinkerAdapter(), pricing=pricing)
    return TestClient(create_app(runtime))


def test_health_and_cost_estimate(tmp_path: Path):
    with make_client(tmp_path) as client:
        assert client.get("/api/health").json()["ok"] is True
        response = client.post(
            "/api/v1/evolutions/estimate", json={"name": "test", "max_usd": 10}
        )
        assert response.status_code == 200
        assert response.json()["reserved_usd"] > 0


def test_create_start_promote_and_chat(tmp_path: Path):
    with make_client(tmp_path) as client:
        created = client.post(
            "/api/v1/evolutions",
            json={
                "name": "API test",
                "max_usd": 5,
                "search": {
                    "population": 2,
                    "generations": 1,
                    "survivors": 1,
                    "steps_per_candidate": 1,
                    "batch_size": 1,
                    "max_sequence_tokens": 64,
                },
                "benchmarks": [{"name": "gsm8k", "weight": 1, "examples": 1}],
            },
        )
        assert created.status_code == 201
        evolution_id = created.json()["id"]

        # Exercise the engine directly so the contract test remains deterministic.
        import asyncio

        asyncio.run(client.app.state.runtime.engine.run(evolution_id))
        detail = client.get(f"/api/v1/evolutions/{evolution_id}").json()
        candidate = detail["candidates"][0]
        promoted = client.post(
            f"/api/v1/evolutions/{evolution_id}/promote",
            json={"candidate_id": candidate["id"]},
        )
        assert promoted.status_code == 200

        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": f"neoform://evolutions/{evolution_id}/champion",
                "messages": [{"role": "user", "content": "2+2"}],
            },
        )
        assert chat.status_code == 200
        assert chat.headers["x-neoform-candidate"] == candidate["id"]

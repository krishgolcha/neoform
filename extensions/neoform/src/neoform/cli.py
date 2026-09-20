from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .domain import EvolutionSpec
from .evaluators import list_evaluators
from .pricing import PricingCatalog
from .runtime import build_runtime
from .settings import Settings

app = typer.Typer(help="NEOFORM — evolutionary post-training powered by Tinker")
console = Console()


def _read_spec(path: Path) -> EvolutionSpec:
    return EvolutionSpec.model_validate_json(path.read_text())


@app.command()
def evaluators() -> None:
    """List bundled evaluators and dataset sizes."""
    table = Table("Name", "Examples", "Description")
    for item in list_evaluators():
        table.add_row(str(item["name"]), str(item["examples"]), str(item["description"]))
    console.print(table)


@app.command()
def doctor() -> None:
    """Check local configuration and Tinker connectivity."""

    async def run() -> None:
        settings = Settings.from_env()
        console.print(f"Data directory: [cyan]{settings.data_dir}[/cyan]")
        try:
            runtime = build_runtime()
            result = await runtime.adapter.doctor()
            console.print(Panel.fit(json.dumps(result, indent=2), title="Tinker connected"))
        except Exception as exc:
            console.print(f"[red]Connection failed:[/red] {exc}")
            raise typer.Exit(1) from exc

    asyncio.run(run())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8787),
    reload: bool = typer.Option(False),
) -> None:
    """Start the NEOFORM API."""
    uvicorn.run("neoform.api:app", host=host, port=port, reload=reload)


@app.command()
def estimate(spec: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Estimate the conservative maximum cost of an evolution."""

    async def run() -> None:
        catalog = await PricingCatalog.fetch()
        result = catalog.estimate(_read_spec(spec))
        console.print_json(result.model_dump_json())

    asyncio.run(run())


@app.command()
def evolve(
    spec: Annotated[Path, typer.Argument(exists=True, readable=True)],
    api: str = typer.Option("http://127.0.0.1:8787"),
) -> None:
    """Create and start an evolution through a running NEOFORM API."""
    with httpx.Client(base_url=api, timeout=30) as client:
        created = client.post(
            "/api/v1/evolutions", json=_read_spec(spec).model_dump()
        ).raise_for_status()
        evolution = created.json()
        client.post(f"/api/v1/evolutions/{evolution['id']}/start").raise_for_status()
        console.print(f"Started [bold cyan]{evolution['id']}[/bold cyan]")


@app.command()
def status(api: str = typer.Option("http://127.0.0.1:8787")) -> None:
    """List evolutions."""
    response = httpx.get(f"{api}/api/v1/evolutions", timeout=10).raise_for_status().json()
    table = Table("ID", "Name", "Status", "Spent", "Champion")
    for item in response["data"]:
        table.add_row(
            item["id"],
            item["name"],
            item["status"],
            f"${item['spent']:.4f}",
            item["champion_id"] or "—",
        )
    console.print(table)


@app.command()
def promote(
    evolution_id: str,
    candidate_id: str,
    api: str = typer.Option("http://127.0.0.1:8787"),
) -> None:
    """Promote a completed candidate to champion."""
    response = httpx.post(
        f"{api}/api/v1/evolutions/{evolution_id}/promote",
        json={"candidate_id": candidate_id},
        timeout=10,
    )
    response.raise_for_status()
    console.print(f"Promoted [bold green]{candidate_id}[/bold green]")


@app.command("export")
def export_evolution(
    evolution_id: str,
    output: Annotated[Path, typer.Option()] = Path("neoform-evolution.json"),
    api: str = typer.Option("http://127.0.0.1:8787"),
) -> None:
    """Export an immutable evolution record."""
    response = httpx.get(
        f"{api}/api/v1/evolutions/{evolution_id}", timeout=10
    ).raise_for_status()
    output.write_text(json.dumps(response.json(), indent=2))
    console.print(f"Wrote [cyan]{output}[/cyan]")


if __name__ == "__main__":
    app()

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import CandidateStatus, Event, EvolutionSpec, EvolutionStatus, Genotype


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._migrate()

    def _migrate(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS evolutions (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, spec TEXT NOT NULL,
                    status TEXT NOT NULL, champion_id TEXT, spent REAL NOT NULL DEFAULT 0,
                    reserved REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY, evolution_id TEXT NOT NULL, generation INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL, parent_id TEXT, genotype TEXT NOT NULL,
                    status TEXT NOT NULL, score REAL, state_checkpoint TEXT,
                    sampler_checkpoint TEXT, metrics TEXT, error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(evolution_id) REFERENCES evolutions(id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, evolution_id TEXT NOT NULL,
                    type TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_evolution_sequence
                    ON events(evolution_id, sequence);
                """
            )

    def create_evolution(self, spec: EvolutionSpec) -> str:
        evolution_id = f"evo_{uuid.uuid4().hex[:12]}"
        timestamp = _now()
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO evolutions VALUES (?, ?, ?, ?, NULL, 0, 0, ?, ?)",
                (
                    evolution_id,
                    spec.name,
                    spec.model_dump_json(),
                    EvolutionStatus.DRAFT,
                    timestamp,
                    timestamp,
                ),
            )
        self.add_event(Event(evolution_id=evolution_id, type="evolution.created"))
        return evolution_id

    def list_evolutions(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM evolutions ORDER BY created_at DESC"
        ).fetchall()
        return [self._evolution(row) for row in rows]

    def get_evolution(self, evolution_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM evolutions WHERE id = ?", (evolution_id,)
        ).fetchone()
        if not row:
            return None
        result = self._evolution(row)
        result["candidates"] = self.list_candidates(evolution_id)
        return result

    def _evolution(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["spec"] = json.loads(result["spec"])
        return result

    def set_evolution_status(self, evolution_id: str, status: EvolutionStatus) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE evolutions SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now(), evolution_id),
            )
        self.add_event(Event(evolution_id=evolution_id, type=f"evolution.{status}"))

    def set_budget(self, evolution_id: str, spent: float, reserved: float) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE evolutions SET spent = ?, reserved = ?, updated_at = ? WHERE id = ?",
                (spent, reserved, _now(), evolution_id),
            )

    def create_candidate(
        self,
        evolution_id: str,
        generation: int,
        ordinal: int,
        genotype: Genotype,
        parent_id: str | None = None,
    ) -> str:
        candidate_id = f"cand_{generation}_{ordinal}_{uuid.uuid4().hex[:7]}"
        timestamp = _now()
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO candidates VALUES
                (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?)""",
                (
                    candidate_id,
                    evolution_id,
                    generation,
                    ordinal,
                    parent_id,
                    genotype.model_dump_json(),
                    CandidateStatus.QUEUED,
                    timestamp,
                    timestamp,
                ),
            )
        self.add_event(
            Event(
                evolution_id=evolution_id,
                type="candidate.created",
                data={
                    "candidate_id": candidate_id,
                    "generation": generation,
                    "parent_id": parent_id,
                },
            )
        )
        return candidate_id

    def update_candidate(self, candidate_id: str, **values: Any) -> None:
        allowed = {
            "status",
            "score",
            "state_checkpoint",
            "sampler_checkpoint",
            "metrics",
            "error",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if "metrics" in updates and not isinstance(updates["metrics"], str):
            updates["metrics"] = json.dumps(updates["metrics"])
        if not updates:
            return
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._lock, self.connection:
            self.connection.execute(
                f"UPDATE candidates SET {assignments} WHERE id = ?",
                (*updates.values(), candidate_id),
            )

    def list_candidates(self, evolution_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM candidates WHERE evolution_id = ? ORDER BY generation, ordinal",
            (evolution_id,),
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["genotype"] = json.loads(item["genotype"])
            item["metrics"] = json.loads(item["metrics"]) if item["metrics"] else None
            results.append(item)
        return results

    def promote(self, evolution_id: str, candidate_id: str) -> None:
        row = self.connection.execute(
            "SELECT status FROM candidates WHERE id = ? AND evolution_id = ?",
            (candidate_id, evolution_id),
        ).fetchone()
        if not row or row["status"] != CandidateStatus.COMPLETE:
            raise ValueError("Only a complete candidate can be promoted")
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE evolutions SET champion_id = ?, updated_at = ? WHERE id = ?",
                (candidate_id, _now(), evolution_id),
            )
        self.add_event(
            Event(
                evolution_id=evolution_id,
                type="candidate.promoted",
                data={"candidate_id": candidate_id},
            )
        )

    def add_event(self, event: Event) -> int:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO events(evolution_id, type, data, created_at) VALUES (?, ?, ?, ?)",
                (
                    event.evolution_id,
                    event.type,
                    json.dumps(event.data),
                    event.created_at.isoformat(),
                ),
            )
        return int(cursor.lastrowid)

    def events(self, evolution_id: str, after: int = 0) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE evolution_id = ? AND sequence > ? ORDER BY sequence",
            (evolution_id, after),
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["data"] = json.loads(item["data"])
            results.append(item)
        return results

    def recover_interrupted(self) -> int:
        interrupted = (
            CandidateStatus.TRAINING,
            CandidateStatus.PROBING,
            CandidateStatus.EVALUATING,
        )
        with self._lock, self.connection:
            cursor = self.connection.execute(
                f"UPDATE candidates SET status = ?, error = ?, updated_at = ? "
                f"WHERE status IN ({','.join('?' for _ in interrupted)})",
                (CandidateStatus.QUEUED, "Recovered after process restart", _now(), *interrupted),
            )
            self.connection.execute(
                "UPDATE evolutions SET status = ?, updated_at = ? WHERE status = ?",
                (EvolutionStatus.PAUSED, _now(), EvolutionStatus.RUNNING),
            )
        return cursor.rowcount

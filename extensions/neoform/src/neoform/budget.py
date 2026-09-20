from __future__ import annotations

import asyncio
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Reservation:
    key: str
    amount: float


class BudgetLedger:
    """Concurrency-safe conservative budget reservation ledger."""

    def __init__(self, maximum: float, spent: float = 0, reserved: float = 0):
        self.maximum = maximum
        self.spent = spent
        self.reserved = reserved
        self._reservations: dict[str, float] = {}
        self._lock = asyncio.Lock()

    @property
    def remaining(self) -> float:
        return max(0.0, self.maximum - self.spent - self.reserved)

    async def reserve(self, key: str, amount: float) -> Reservation:
        async with self._lock:
            if key in self._reservations:
                return Reservation(key, self._reservations[key])
            if amount > self.remaining + 1e-9:
                raise BudgetExceeded(
                    f"${amount:.4f} reservation exceeds ${self.remaining:.4f} remaining"
                )
            self._reservations[key] = amount
            self.reserved += amount
            return Reservation(key, amount)

    async def reconcile(self, reservation: Reservation, actual: float) -> None:
        async with self._lock:
            reserved = self._reservations.pop(reservation.key, 0)
            self.reserved = max(0.0, self.reserved - reserved)
            self.spent += max(0.0, actual)

    async def release(self, reservation: Reservation) -> None:
        async with self._lock:
            reserved = self._reservations.pop(reservation.key, 0)
            self.reserved = max(0.0, self.reserved - reserved)


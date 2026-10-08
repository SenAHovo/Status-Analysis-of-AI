"""Task and call budget ledger with atomic reserve/settle semantics.

Estimates and actuals are kept separate: reservations mark estimated usage,
settlements record what the provider reported, and failed calls whose usage is
unknown keep their reservation as unknown consumption so they are never
treated as free. A thread lock keeps reservations atomic for later parallel
search nodes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

RUN = "run"
WINDOW = "window"
OUTPUT = "output"
SEARCH_REQUESTS = "search_requests"
RETRIEVAL_REQUESTS = "retrieval_requests"

# A root-less client is reserved for isolated unit tests. Real application
# entry points must create a run-scoped ledger from ``config/budgets.yaml``.
_DEFAULT_CAPACITIES: dict[str, int] = {
    RUN: 300_000,
    WINDOW: 16_000,
    OUTPUT: 2_000,
    SEARCH_REQUESTS: 30,
    RETRIEVAL_REQUESTS: 60,
}


def _validate_amount(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


@dataclass(frozen=True)
class BudgetLimits:
    capacities: dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_CAPACITIES))

    def __post_init__(self) -> None:
        for kind, capacity in self.capacities.items():
            _validate_amount(capacity, f"capacity for {kind}")

    def capacity(self, kind: str) -> int:
        try:
            return self.capacities[kind]
        except KeyError as exc:
            raise KeyError(f"unknown budget kind: {kind}") from exc


class BudgetExceeded(RuntimeError):
    """Budget exhausted for a kind; safe reason only."""


class TokenLedgerStateError(ValueError):
    """A persisted ledger snapshot cannot be safely recovered."""


class TokenLedger:
    """One ledger per run; shared by parallel workers via a state snapshot."""

    def __init__(self, run_id: str, limits: BudgetLimits | None = None):
        self.run_id = run_id
        self.limits = limits or BudgetLimits()
        self.estimated: dict[str, int] = {}
        self.actual: dict[str, int] = {}
        self.unknown: dict[str, int] = {}
        self._reservations: dict[str, tuple[str, int]] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def _ticket(self) -> str:
        self._counter += 1
        return f"rsv-{self.run_id}-{self._counter}"

    def reserve(self, kind: str, amount: int) -> str:
        """Atomically reserve estimated usage; raises BudgetExceeded."""
        _validate_amount(amount, "reservation amount")
        with self._lock:
            if not self._available_locked(kind, amount):
                raise BudgetExceeded(kind)
            token = self._ticket()
            self._reservations[token] = (kind, amount)
            self.estimated[kind] = self.estimated.get(kind, 0) + amount
            return token

    def settle(self, token: str, actual_amount: int | None) -> None:
        """Settle a reservation with provider actuals (None means unknown)."""
        with self._lock:
            try:
                kind, estimate = self._reservations.pop(token)
            except KeyError as exc:
                raise KeyError("unknown reservation") from exc
            if actual_amount is None:
                self.unknown[kind] = self.unknown.get(kind, 0) + estimate
            else:
                _validate_amount(actual_amount, "actual amount")
                self.actual[kind] = self.actual.get(kind, 0) + actual_amount

    def cancel(self, token: str) -> None:
        """Release an unused reservation."""
        with self._lock:
            try:
                kind, estimate = self._reservations.pop(token)
            except KeyError as exc:
                raise KeyError("unknown reservation") from exc
            self.estimated[kind] = max(0, self.estimated.get(kind, 0) - estimate)

    def meter(self, kind: str, amount: int = 1) -> bool:
        """Record a non-token metered action after an availability check."""
        _validate_amount(amount, "meter amount")
        with self._lock:
            if not self._available_locked(kind, amount):
                raise BudgetExceeded(kind)
            self.actual[kind] = self.actual.get(kind, 0) + amount
            return True

    def check(self, kind: str, amount: int = 1) -> bool:
        """True when a reservation of ``amount`` for ``kind`` fits the limit."""
        _validate_amount(amount, "check amount")
        with self._lock:
            return self._available_locked(kind, amount)

    def remaining(self, kind: str) -> int:
        with self._lock:
            capacity = self.limits.capacity(kind)
            used = self._used_locked(kind)
            return max(0, capacity - used)

    def _used_locked(self, kind: str) -> int:
        in_flight = sum(
            amount for reservation_kind, amount in self._reservations.values()
            if reservation_kind == kind
        )
        return (
            self.actual.get(kind, 0)
            + self.unknown.get(kind, 0)
            + in_flight
        )

    def _available_locked(self, kind: str, amount: int) -> bool:
        capacity = self.limits.capacity(kind)
        return capacity - self._used_locked(kind) >= amount

    def snapshot(self) -> dict:
        """Plain state for persistence; lock is not serialized."""
        with self._lock:
            return {
                "run_id": self.run_id,
                "capacities": dict(self.limits.capacities),
                "estimated": dict(self.estimated),
                "actual": dict(self.actual),
                "unknown": dict(self.unknown),
                "reservations": dict(self._reservations),
                "counter": self._counter,
            }

    @classmethod
    def from_snapshot(cls, snapshot: dict, *, recover_in_flight: bool = True) -> TokenLedger:
        """Restore only a complete, typed snapshot.

        Recovery callers may conservatively convert in-flight reservations to
        unknown consumption. A live persistent ledger reloads with
        ``recover_in_flight=False`` so it can settle its own reservation after
        an inter-process state refresh.
        """

        if not isinstance(snapshot, dict):
            raise TokenLedgerStateError("token_ledger_snapshot_invalid")
        run_id = snapshot.get("run_id")
        capacities = snapshot.get("capacities")
        if not isinstance(run_id, str) or not run_id or not isinstance(capacities, dict):
            raise TokenLedgerStateError("token_ledger_snapshot_invalid")
        try:
            ledger = cls(run_id, BudgetLimits(dict(capacities)))
            allowed = set(ledger.limits.capacities)
            counters: dict[str, dict[str, int]] = {}
            for name in ("estimated", "actual", "unknown"):
                value = snapshot.get(name)
                if not isinstance(value, dict) or any(
                    not isinstance(kind, str)
                    or kind not in allowed
                    or type(amount) is not int
                    or amount < 0
                    for kind, amount in value.items()
                ):
                    raise TokenLedgerStateError("token_ledger_snapshot_invalid")
                counters[name] = dict(value)
            raw_reservations = snapshot.get("reservations")
            if not isinstance(raw_reservations, dict):
                raise TokenLedgerStateError("token_ledger_snapshot_invalid")
            reservations: dict[str, tuple[str, int]] = {}
            for token, value in raw_reservations.items():
                if (
                    not isinstance(token, str)
                    or not isinstance(value, (list, tuple))
                    or len(value) != 2
                    or not isinstance(value[0], str)
                    or value[0] not in allowed
                    or type(value[1]) is not int
                    or value[1] < 0
                ):
                    raise TokenLedgerStateError("token_ledger_snapshot_invalid")
                reservations[token] = (value[0], value[1])
            counter = snapshot.get("counter")
            _validate_amount(counter, "snapshot counter")
        except (KeyError, TypeError, ValueError) as exc:
            raise TokenLedgerStateError("token_ledger_snapshot_invalid") from exc

        ledger.estimated = counters["estimated"]
        ledger.actual = counters["actual"]
        ledger.unknown = counters["unknown"]
        if recover_in_flight:
            for kind, amount in reservations.values():
                ledger.unknown[kind] = ledger.unknown.get(kind, 0) + amount
            ledger._reservations = {}
        else:
            ledger._reservations = reservations
        ledger._counter = counter
        return ledger

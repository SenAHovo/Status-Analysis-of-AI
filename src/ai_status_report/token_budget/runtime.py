"""Process-safe, run-scoped TokenLedger persistence for paid model calls."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ai_status_report.token_budget.allocator import (
    BudgetLimits,
    TokenLedger,
    TokenLedgerStateError,
)
from ai_status_report.token_budget.config import load_budget_limits


@contextmanager
def _run_file_lock(path: Path):
    """Hold an advisory cross-process lock beside one run ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PersistentTokenLedger(TokenLedger):
    """A ledger whose every mutation reloads and atomically persists one run."""

    def __init__(self, run_id: str, path: Path, *, limits: BudgetLimits):
        super().__init__(run_id, limits)
        self.path = path
        self.lock_path = path.with_suffix(".lock")
        with _run_file_lock(self.lock_path):
            self._reload_locked()

    def _replace_state(self, restored: TokenLedger) -> None:
        self.limits = restored.limits
        self.estimated = restored.estimated
        self.actual = restored.actual
        self.unknown = restored.unknown
        self._reservations = restored._reservations
        self._counter = restored._counter

    def _reload_locked(self) -> None:
        if not self.path.is_file():
            return
        try:
            snapshot = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TokenLedgerStateError("token_ledger_snapshot_invalid") from exc
        restored = TokenLedger.from_snapshot(snapshot, recover_in_flight=False)
        if restored.run_id != self.run_id:
            raise TokenLedgerStateError("token_ledger_snapshot_invalid")
        self._replace_state(restored)

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(self.snapshot(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def reserve(self, kind: str, amount: int) -> str:
        with _run_file_lock(self.lock_path):
            self._reload_locked()
            token = super().reserve(kind, amount)
            self._persist_locked()
            return token

    def settle(self, token: str, actual_amount: int | None) -> None:
        with _run_file_lock(self.lock_path):
            self._reload_locked()
            super().settle(token, actual_amount)
            self._persist_locked()

    def cancel(self, token: str) -> None:
        with _run_file_lock(self.lock_path):
            self._reload_locked()
            super().cancel(token)
            self._persist_locked()

    def meter(self, kind: str, amount: int = 1) -> bool:
        with _run_file_lock(self.lock_path):
            self._reload_locked()
            result = super().meter(kind, amount)
            self._persist_locked()
            return result


def new_run_ledger(run_id: str, *, root: Path | None = None) -> TokenLedger:
    """Create one ledger for one workflow run without an unaccounted fallback."""

    if root is None:
        return TokenLedger(run_id)
    from ai_status_report.storage.search_results import run_directory

    path = run_directory(root, run_id) / "token_usage.json"
    return PersistentTokenLedger(run_id, path, limits=load_budget_limits(root))

"""Lifecycle management for the local Chroma and A2A teaching services.

This module manages infrastructure processes only.  It does not participate in
LangGraph routing or replace the controller's A2A client calls.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import httpx


class LocalServiceError(RuntimeError):
    """A stable, safe error raised while preparing a local teaching service."""


@dataclass(frozen=True)
class LocalServiceSpec:
    """One local service required by the teaching command."""

    name: str
    host: str
    port: int
    command: tuple[str, ...]
    health_url: str
    expected_agent_name: str | None = None


@dataclass
class ManagedLocalService:
    """The observable result of one service health check or start attempt."""

    spec: LocalServiceSpec
    log_path: Path
    reused: bool
    process: Any | None = None

    @property
    def started_by_manager(self) -> bool:
        return self.process is not None


def _session_id(now: datetime | None = None) -> str:
    value = now or datetime.now(UTC)
    return value.strftime("teaching-%Y%m%dT%H%M%S%fZ")


class LocalServiceManager:
    """Ensure the three existing local runtime services are usable.

    Chroma is infrastructure rather than a fourth Agent.  The two Uvicorn
    processes expose the existing A2A applications unchanged.

    Chroma's current CLI documents ``run --path --host --port``; Uvicorn
    documents the ``module:attribute --factory`` command shape.
    Sources: https://docs.trychroma.com/docs/run-chroma/client-server and
    https://www.uvicorn.org/settings/.
    """

    def __init__(
        self,
        root: Path,
        *,
        session_id: str | None = None,
        startup_timeout_seconds: float = 15.0,
        graceful_shutdown_timeout_seconds: float = 15.0,
        retry_interval_seconds: float = 0.25,
        process_factory: Callable[..., Any] = subprocess.Popen,
        health_probe: Callable[[LocalServiceSpec], bool] | None = None,
        port_open: Callable[[str, int], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = root.resolve()
        self.session_id = session_id or _session_id()
        self.startup_timeout_seconds = startup_timeout_seconds
        self.graceful_shutdown_timeout_seconds = graceful_shutdown_timeout_seconds
        self.retry_interval_seconds = retry_interval_seconds
        self._process_factory = process_factory
        self._health_probe = health_probe or self._default_health_probe
        self._port_open = port_open or self._default_port_open
        self._sleep = sleep
        self._managed: list[ManagedLocalService] = []

    @property
    def log_directory(self) -> Path:
        return self.root / "data" / "teaching_sessions" / self.session_id / "services"

    def ensure_services(self) -> list[ManagedLocalService]:
        """Reuse healthy services or launch all missing teaching services.

        A port occupied by an unhealthy or unrelated process is never killed.
        Any services launched before a later failure are cleaned up before the
        exception is returned.
        """

        try:
            return [self.ensure_service(spec) for spec in self.service_specs()]
        except Exception:
            self.stop()
            raise

    def ensure_service(self, spec: LocalServiceSpec) -> ManagedLocalService:
        log_path = self.log_directory / f"{spec.name}.log"
        if self._health_probe(spec):
            return ManagedLocalService(spec=spec, log_path=log_path, reused=True)
        if self._port_open(spec.host, spec.port):
            raise LocalServiceError(f"service_port_in_use:{spec.name}")
        executable = Path(spec.command[0])
        if not executable.is_file() and shutil.which(spec.command[0]) is None:
            raise LocalServiceError(f"service_executable_missing:{spec.name}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("ab")
        try:
            process_kwargs: dict[str, object] = {
                "cwd": self.root,
                "env": os.environ.copy(),
                "stdout": handle,
                "stderr": subprocess.STDOUT,
            }
            # Python documents that CTRL_BREAK_EVENT requires a new process group
            # on Windows. This lets Chroma finish its persistent-index shutdown.
            # Source: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.send_signal
            if os.name == "nt":
                process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            process = self._process_factory(
                list(spec.command),
                **process_kwargs,
            )
        except OSError as exc:
            handle.close()
            raise LocalServiceError(f"service_start_failed:{spec.name}") from exc
        handle.close()
        managed = ManagedLocalService(spec=spec, log_path=log_path, reused=False, process=process)
        self._managed.append(managed)
        self._wait_until_healthy(managed)
        return managed

    def stop(self) -> None:
        """Stop only child processes created by this manager, in reverse order."""

        for managed in reversed(self._managed):
            process = managed.process
            if process is None or process.poll() is not None:
                continue
            self._request_graceful_stop(process)
            try:
                process.wait(timeout=self.graceful_shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._managed.clear()

    @staticmethod
    def _request_graceful_stop(process: Any) -> None:
        """Request normal service shutdown before the hard-stop fallback.

        ``Popen.terminate()`` maps to ``TerminateProcess()`` on Windows, while
        ``CTRL_BREAK_EVENT`` can be sent to a child created in a new process
        group. The latter gives Chroma a chance to finish HNSW persistence.
        Source: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.terminate
        """

        if os.name == "nt":
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                return
            except (AttributeError, OSError, ValueError):
                pass
        process.terminate()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def service_specs(self) -> tuple[LocalServiceSpec, ...]:
        chroma_executable = self._venv_executable("chroma")
        python = sys.executable
        chroma_path = self.root / "data" / "vector_store" / "chroma"
        return (
            LocalServiceSpec(
                name="chroma",
                host="127.0.0.1",
                port=8000,
                command=(
                    str(chroma_executable),
                    "run",
                    "--path",
                    str(chroma_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ),
                health_url="http://127.0.0.1:8000/api/v2/heartbeat",
            ),
            LocalServiceSpec(
                name="search-agent",
                host="127.0.0.1",
                port=8001,
                command=(
                    python,
                    "-m",
                    "uvicorn",
                    "ai_status_report.a2a.apps:network_search_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8001",
                ),
                health_url="http://127.0.0.1:8001/.well-known/agent-card.json",
                expected_agent_name="Network Search Agent",
            ),
            LocalServiceSpec(
                name="document-agent",
                host="127.0.0.1",
                port=8002,
                command=(
                    python,
                    "-m",
                    "uvicorn",
                    "ai_status_report.a2a.apps:document_generation_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8002",
                ),
                health_url="http://127.0.0.1:8002/.well-known/agent-card.json",
                expected_agent_name="Document Generation Agent",
            ),
        )

    def _venv_executable(self, name: str) -> Path:
        suffix = ".exe" if os.name == "nt" else ""
        return Path(sys.executable).parent / f"{name}{suffix}"

    def _wait_until_healthy(self, managed: ManagedLocalService) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if managed.process is not None and managed.process.poll() is not None:
                raise LocalServiceError(f"service_start_failed:{managed.spec.name}")
            if self._health_probe(managed.spec):
                return
            self._sleep(self.retry_interval_seconds)
        raise LocalServiceError(f"service_start_timeout:{managed.spec.name}")

    @staticmethod
    def _default_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            return False

    @staticmethod
    def _default_health_probe(spec: LocalServiceSpec) -> bool:
        try:
            with httpx.Client(timeout=0.5, trust_env=False) as client:
                response = client.get(spec.health_url)
            if response.status_code != 200:
                return False
            if spec.expected_agent_name is None:
                return True
            payload = response.json()
            return isinstance(payload, dict) and payload.get("name") == spec.expected_agent_name
        except (httpx.HTTPError, ValueError):
            return False

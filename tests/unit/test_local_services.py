import os
import signal
import subprocess
import sys

import pytest

from ai_status_report.harness.local_services import LocalServiceError, LocalServiceManager


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.signals = []
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def send_signal(self, value):
        self.signals.append(value)
        self.returncode = 0

    def wait(self, timeout):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = 0


def test_manager_starts_missing_services_writes_logs_and_stops_owned_processes(tmp_path):
    commands = []
    probes = {"chroma": [False, True], "search-agent": [False, True], "document-agent": [False, True]}

    def process_factory(command, **kwargs):
        commands.append((command, kwargs))
        return FakeProcess()

    manager = LocalServiceManager(
        tmp_path,
        session_id="teaching-test",
        process_factory=process_factory,
        health_probe=lambda spec: probes[spec.name].pop(0),
        port_open=lambda *_: False,
        sleep=lambda _: None,
    )

    services = manager.ensure_services()

    assert [service.spec.name for service in services] == ["chroma", "search-agent", "document-agent"]
    assert all(service.started_by_manager for service in services)
    assert all(service.log_path.is_file() for service in services)
    assert commands[0][0][1:3] == ["run", "--path"]
    assert commands[1][0][:4] == [sys.executable, "-m", "uvicorn", "ai_status_report.a2a.apps:network_search_app"]
    assert all(kwargs["cwd"] == tmp_path.resolve() for _, kwargs in commands)
    if os.name == "nt":
        assert all(kwargs["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP for _, kwargs in commands)

    manager.stop()

    if os.name == "nt":
        assert all(service.process.signals == [signal.CTRL_BREAK_EVENT] for service in services)
    else:
        assert all(service.process.terminated for service in services)


def test_manager_reuses_healthy_services_without_starting_or_stopping_them(tmp_path):
    manager = LocalServiceManager(
        tmp_path,
        session_id="teaching-test",
        process_factory=lambda *_args, **_kwargs: pytest.fail("must not start a healthy service"),
        health_probe=lambda _spec: True,
        port_open=lambda *_: pytest.fail("must not inspect a healthy service port"),
    )

    services = manager.ensure_services()
    manager.stop()

    assert all(service.reused for service in services)
    assert all(service.process is None for service in services)


def test_manager_refuses_to_touch_an_unhealthy_occupied_port(tmp_path):
    manager = LocalServiceManager(
        tmp_path,
        session_id="teaching-test",
        process_factory=lambda *_args, **_kwargs: pytest.fail("must not replace an occupied port"),
        health_probe=lambda _spec: False,
        port_open=lambda _host, port: port == 8000,
    )

    with pytest.raises(LocalServiceError, match="service_port_in_use:chroma"):
        manager.ensure_services()

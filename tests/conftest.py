"""Shared fixtures for SDK tests."""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Generator

import pytest

from spectracost.transport import _shutdown_all_active_transports


@pytest.fixture(autouse=True)
def _isolate_transports() -> Generator[None, None, None]:
    """Shut down every Transport created during a test case.

    HTTPServer uses SO_REUSEADDR, so the ephemeral port assigned to
    one test's fixture can be reassigned to the next. A prior test's
    still-running Transport would then retry POSTs to the next test's
    server and leak events with the wrong attribution. Tearing down
    the transports between tests stops that.
    """
    yield
    _shutdown_all_active_transports()


class _TelemetryHandler(BaseHTTPRequestHandler):
    """Captures events POSTed to /v1/events."""

    server: "_TelemetryServer"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        events = json.loads(body)
        self.server.received_events.extend(events)
        self.send_response(202)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass  # Silence request logs in test output


class _TelemetryServer(HTTPServer):
    received_events: list[dict]


@pytest.fixture
def telemetry_server() -> Generator[_TelemetryServer, None, None]:
    """Start a local HTTP server that captures telemetry events."""
    server = _TelemetryServer(("127.0.0.1", 0), _TelemetryHandler)
    server.received_events = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


@pytest.fixture
def telemetry_url(telemetry_server: _TelemetryServer) -> str:
    """The base URL for the telemetry server."""
    host, port = telemetry_server.server_address
    return f"http://{host}:{port}"

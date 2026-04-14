"""Tests for the async transport layer."""

from __future__ import annotations

import time


from spectracost.transport import Transport
from spectracost.types import UsageEvent


def _make_event(**overrides) -> UsageEvent:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        endpoint="chat.completions",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        latency_ms=500,
        status="success",
    )
    defaults.update(overrides)
    return UsageEvent(**defaults)


def test_transport_sends_events(telemetry_server, telemetry_url: str) -> None:
    """Transport sends enqueued events to the telemetry server."""
    transport = Transport(endpoint=telemetry_url, api_key="sprc_test")

    transport.enqueue(_make_event())
    transport.enqueue(_make_event(model="gpt-4o-mini"))

    time.sleep(2)

    assert len(telemetry_server.received_events) >= 2
    models = {e["model"] for e in telemetry_server.received_events}
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models

    transport.flush()


def test_transport_flush_sends_remaining(telemetry_server, telemetry_url: str) -> None:
    """flush() sends all remaining events before returning."""
    transport = Transport(endpoint=telemetry_url, api_key="sprc_test")

    for i in range(5):
        transport.enqueue(_make_event(model=f"model-{i}"))

    transport.flush()

    assert len(telemetry_server.received_events) == 5


def test_transport_survives_dead_endpoint() -> None:
    """Transport doesn't crash when the endpoint is unreachable."""
    transport = Transport(endpoint="http://127.0.0.1:1", api_key="sprc_test")

    transport.enqueue(_make_event())

    # Give it a moment to try and fail
    time.sleep(1)

    transport.flush()
    # No exception = success

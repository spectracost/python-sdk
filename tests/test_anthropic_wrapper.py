"""Tests for the Anthropic provider wrapper."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from spectracost import instrument


def _make_anthropic_client() -> MagicMock:
    """Create a mock that looks like an Anthropic client.

    Anthropic clients have `messages` but NOT `chat`. We use spec to prevent
    MagicMock from auto-creating a `chat` attribute, which would cause duck
    typing to misidentify this as an OpenAI client.
    """
    class _FakeAnthropic:
        messages: object
    client = MagicMock(spec=_FakeAnthropic)
    # Re-add messages as a full MagicMock so we can set return_value etc.
    client.messages = MagicMock()
    return client


def _make_message_response(
    input_tokens: int = 200, output_tokens: int = 80
) -> MagicMock:
    resp = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.usage.cache_read_input_tokens = 0
    return resp


def test_messages_create_returns_same_response(telemetry_url: str) -> None:
    """Wrapped messages.create returns the same response as the original."""
    client = _make_anthropic_client()
    expected = _make_message_response()
    client.messages.create.return_value = expected

    wrapped = instrument(client, api_key="sprc_test", endpoint=telemetry_url)
    result = wrapped.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[])

    assert result is expected


def test_messages_create_emits_telemetry(
    telemetry_server, telemetry_url: str
) -> None:
    """Wrapped messages.create sends a telemetry event with correct fields."""
    client = _make_anthropic_client()
    client.messages.create.return_value = _make_message_response(200, 80)

    wrapped = instrument(
        client,
        api_key="sprc_test",
        endpoint=telemetry_url,
        team="support",
        service="ticket-classifier",
    )
    wrapped.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[])

    time.sleep(2)

    assert len(telemetry_server.received_events) >= 1
    event = telemetry_server.received_events[0]
    assert event["provider"] == "anthropic"
    assert event["model"] == "claude-sonnet-4-20250514"
    assert event["endpoint"] == "messages"
    assert event["input_tokens"] == 200
    assert event["output_tokens"] == 80
    assert event["team"] == "support"
    assert event["service"] == "ticket-classifier"


def test_messages_create_error_still_raises(telemetry_url: str) -> None:
    """If the underlying API call raises, the wrapper re-raises."""
    client = _make_anthropic_client()
    client.messages.create.side_effect = ConnectionError("timeout")

    wrapped = instrument(client, api_key="sprc_test", endpoint=telemetry_url)

    with pytest.raises(ConnectionError, match="timeout"):
        wrapped.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[])


def test_telemetry_failure_does_not_affect_response() -> None:
    """If the telemetry endpoint is unreachable, LLM calls still succeed."""
    client = _make_anthropic_client()
    expected = _make_message_response()
    client.messages.create.return_value = expected

    wrapped = instrument(
        client, api_key="sprc_test", endpoint="http://127.0.0.1:1"
    )
    result = wrapped.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[])

    assert result is expected

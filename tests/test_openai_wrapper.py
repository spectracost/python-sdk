"""Tests for the OpenAI provider wrapper."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from spectracost import instrument


def _make_openai_client() -> MagicMock:
    """Create a mock that looks like an OpenAI client."""
    client = MagicMock()
    client.__class__ = type("OpenAI", (), {"__module__": "openai._client"})
    return client


def _make_completion_response(
    input_tokens: int = 100, output_tokens: int = 50
) -> MagicMock:
    resp = MagicMock()
    resp.usage.prompt_tokens = input_tokens
    resp.usage.completion_tokens = output_tokens
    resp.usage.prompt_tokens_details = None
    return resp


def test_chat_completion_returns_same_response(telemetry_url: str) -> None:
    """Wrapped chat.completions.create returns the same response as the original."""
    client = _make_openai_client()
    expected = _make_completion_response()
    client.chat.completions.create.return_value = expected

    wrapped = instrument(client, api_key="sprc_test", endpoint=telemetry_url)
    result = wrapped.chat.completions.create(model="gpt-4o", messages=[])

    assert result is expected


def test_chat_completion_emits_telemetry(
    telemetry_server, telemetry_url: str
) -> None:
    """Wrapped chat.completions.create sends a telemetry event."""
    client = _make_openai_client()
    client.chat.completions.create.return_value = _make_completion_response(100, 50)

    wrapped = instrument(
        client,
        api_key="sprc_test",
        endpoint=telemetry_url,
        team="search",
        service="query-rewriter",
    )
    wrapped.chat.completions.create(model="gpt-4o", messages=[])

    # Give the background transport time to flush
    time.sleep(2)

    assert len(telemetry_server.received_events) >= 1
    event = telemetry_server.received_events[0]
    assert event["provider"] == "openai"
    assert event["model"] == "gpt-4o"
    assert event["endpoint"] == "chat.completions"
    assert event["input_tokens"] == 100
    assert event["output_tokens"] == 50
    assert event["team"] == "search"
    assert event["service"] == "query-rewriter"
    assert event["status"] == "success"


def test_chat_completion_error_still_raises(telemetry_url: str) -> None:
    """If the underlying API call raises, the wrapper re-raises the same exception."""
    client = _make_openai_client()
    client.chat.completions.create.side_effect = RuntimeError("API error")

    wrapped = instrument(client, api_key="sprc_test", endpoint=telemetry_url)

    with pytest.raises(RuntimeError, match="API error"):
        wrapped.chat.completions.create(model="gpt-4o", messages=[])


def test_chat_completion_error_emits_error_event(
    telemetry_server, telemetry_url: str
) -> None:
    """Failed API calls still emit a telemetry event with status=error."""
    client = _make_openai_client()
    client.chat.completions.create.side_effect = RuntimeError("API error")

    wrapped = instrument(client, api_key="sprc_test", endpoint=telemetry_url)

    with pytest.raises(RuntimeError):
        wrapped.chat.completions.create(model="gpt-4o", messages=[])

    time.sleep(2)

    assert len(telemetry_server.received_events) >= 1
    event = telemetry_server.received_events[0]
    assert event["status"] == "error"
    assert event["error_code"] == "RuntimeError"


def test_embeddings_returns_same_response(telemetry_url: str) -> None:
    """Wrapped embeddings.create returns the same response as the original."""
    client = _make_openai_client()
    expected = MagicMock()
    expected.usage.prompt_tokens = 10
    client.embeddings.create.return_value = expected

    wrapped = instrument(client, api_key="sprc_test", endpoint=telemetry_url)
    result = wrapped.embeddings.create(model="text-embedding-3-small", input="hello")

    assert result is expected


def test_streaming_passes_through_chunks(telemetry_url: str) -> None:
    """Streaming responses yield the same chunks as the original."""
    client = _make_openai_client()

    chunk1 = MagicMock()
    chunk1.usage = None
    chunk2 = MagicMock()
    chunk2.usage = None
    final_chunk = MagicMock()
    final_chunk.usage = MagicMock()
    final_chunk.usage.prompt_tokens = 80
    final_chunk.usage.completion_tokens = 30

    client.chat.completions.create.return_value = iter([chunk1, chunk2, final_chunk])

    wrapped = instrument(client, api_key="sprc_test", endpoint=telemetry_url)
    chunks = list(wrapped.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    assert len(chunks) == 3
    assert chunks[0] is chunk1
    assert chunks[1] is chunk2
    assert chunks[2] is final_chunk


def test_telemetry_failure_does_not_affect_response(telemetry_url: str) -> None:
    """If the telemetry endpoint is unreachable, LLM calls still succeed."""
    client = _make_openai_client()
    expected = _make_completion_response()
    client.chat.completions.create.return_value = expected

    # Point at a dead endpoint
    wrapped = instrument(
        client, api_key="sprc_test", endpoint="http://127.0.0.1:1"
    )
    result = wrapped.chat.completions.create(model="gpt-4o", messages=[])

    assert result is expected

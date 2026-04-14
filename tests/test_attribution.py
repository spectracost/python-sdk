"""Tests for the attribution context manager."""

from __future__ import annotations

import time
from unittest.mock import MagicMock


from spectracost import instrument, attribution
from spectracost.attribution import get_current_attribution


def test_attribution_sets_context() -> None:
    """attribution() context manager sets tags that get_current_attribution() returns."""
    assert get_current_attribution() == {}

    with attribution(team="search", feature="semantic"):
        ctx = get_current_attribution()
        assert ctx["team"] == "search"
        assert ctx["feature"] == "semantic"

    assert get_current_attribution() == {}


def test_attribution_nesting() -> None:
    """Nested attribution contexts merge, with inner taking precedence."""
    with attribution(team="search", feature="a"):
        with attribution(feature="b", customer_id="cust_1"):
            ctx = get_current_attribution()
            assert ctx["team"] == "search"        # from outer
            assert ctx["feature"] == "b"           # overridden by inner
            assert ctx["customer_id"] == "cust_1"  # from inner

        ctx = get_current_attribution()
        assert ctx["feature"] == "a"  # restored to outer
        assert "customer_id" not in ctx


def test_attribution_extra_tags() -> None:
    """Extra keyword arguments are passed through as custom tags."""
    with attribution(team="search", deployment="canary"):
        ctx = get_current_attribution()
        assert ctx["team"] == "search"
        assert ctx["deployment"] == "canary"


def test_attribution_overrides_client_defaults(
    telemetry_server, telemetry_url: str
) -> None:
    """Attribution context overrides client-level defaults on emitted events."""
    client = MagicMock()
    client.__class__ = type("OpenAI", (), {"__module__": "openai._client"})
    resp = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.prompt_tokens_details = None
    client.chat.completions.create.return_value = resp

    wrapped = instrument(
        client,
        api_key="sprc_test",
        endpoint=telemetry_url,
        team="default-team",
        service="default-service",
    )

    with attribution(team="override-team", feature="special-feature"):
        wrapped.chat.completions.create(model="gpt-4o-mini", messages=[])

    time.sleep(2)

    assert len(telemetry_server.received_events) >= 1
    event = telemetry_server.received_events[0]
    assert event["team"] == "override-team"
    assert event["service"] == "default-service"  # not overridden
    assert event["feature"] == "special-feature"

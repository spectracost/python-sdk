"""Tests for the core instrument() function."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from spectracost import instrument


def test_instrument_openai_client(telemetry_url: str) -> None:
    """instrument() wraps an OpenAI-like client successfully."""
    mock_client = MagicMock()
    mock_client.__class__.__module__ = "openai._client"

    wrapped = instrument(mock_client, api_key="sprc_test", endpoint=telemetry_url)

    assert hasattr(wrapped, "chat")
    assert hasattr(wrapped, "embeddings")


def test_instrument_anthropic_client(telemetry_url: str) -> None:
    """instrument() wraps an Anthropic-like client successfully."""
    mock_client = MagicMock()
    mock_client.__class__.__module__ = "anthropic._client"

    wrapped = instrument(mock_client, api_key="sprc_test", endpoint=telemetry_url)

    assert hasattr(wrapped, "messages")


def test_instrument_unsupported_client(telemetry_url: str) -> None:
    """instrument() raises ValueError for unsupported client types."""
    # Use a real object (not MagicMock, which auto-creates any attribute via __getattr__)
    mock_client = type("SomeClient", (), {"__module__": "some_other_library"})()

    with pytest.raises(ValueError, match="Unsupported client type"):
        instrument(mock_client, api_key="sprc_test", endpoint=telemetry_url)


def test_wrapped_client_proxies_unknown_attributes(telemetry_url: str) -> None:
    """Attributes not intercepted by the wrapper are proxied to the underlying client."""
    mock_client = MagicMock()
    mock_client.__class__.__module__ = "openai._client"
    mock_client.some_custom_property = "hello"

    wrapped = instrument(mock_client, api_key="sprc_test", endpoint=telemetry_url)

    assert wrapped.some_custom_property == "hello"


def test_provider_detection_from_host(telemetry_url: str) -> None:
    """An OpenAI client pointing at Groq is auto-detected as groq."""
    mock_client = MagicMock()
    mock_client.__class__.__module__ = "openai._client"
    mock_client.base_url = "https://api.groq.com/openai/v1"

    wrapped = instrument(mock_client, api_key="sprc_test", endpoint=telemetry_url)
    assert wrapped._provider_name == "groq"


def test_provider_detection_from_spectracost_proxy_path(telemetry_url: str) -> None:
    """Clients pointed at Spectracost's proxy parse the provider from the path."""
    mock_client = MagicMock()
    mock_client.__class__.__module__ = "openai._client"
    mock_client.base_url = "https://spectracost.com/proxy/v1/deepseek/v1"

    wrapped = instrument(mock_client, api_key="sprc_test", endpoint=telemetry_url)
    assert wrapped._provider_name == "deepseek"


def test_provider_detection_explicit_override_wins(telemetry_url: str) -> None:
    """instrument(provider=...) overrides auto-detection."""
    mock_client = MagicMock()
    mock_client.__class__.__module__ = "openai._client"
    mock_client.base_url = "https://api.groq.com/openai/v1"

    wrapped = instrument(
        mock_client,
        api_key="sprc_test",
        endpoint=telemetry_url,
        provider="custom-groq",
    )
    assert wrapped._provider_name == "custom-groq"


def test_provider_detection_falls_back_to_class(telemetry_url: str) -> None:
    """Without base_url, detection falls back to the Python class module."""
    mock_client = MagicMock()
    mock_client.__class__.__module__ = "anthropic._client"
    mock_client.base_url = None

    wrapped = instrument(mock_client, api_key="sprc_test", endpoint=telemetry_url)
    assert wrapped._provider_name == "anthropic"


def test_provider_detection_all_known_hosts(telemetry_url: str) -> None:
    """Every host in the detection map resolves to its provider name."""
    cases = {
        "https://api.openai.com/v1": "openai",
        "https://api.anthropic.com": "anthropic",
        "https://api.deepseek.com/v1": "deepseek",
        "https://api.groq.com/openai/v1": "groq",
        "https://api.together.xyz/v1": "together",
        "https://api.mistral.ai/v1": "mistral",
        "https://api.x.ai/v1": "xai",
        "https://openrouter.ai/api/v1": "openrouter",
        "https://api.fireworks.ai/inference/v1": "fireworks",
        "https://generativelanguage.googleapis.com": "google",
        "https://api.cohere.com": "cohere",
    }
    for url, expected in cases.items():
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "openai._client"
        mock_client.base_url = url
        wrapped = instrument(mock_client, api_key="sprc_test", endpoint=telemetry_url)
        assert wrapped._provider_name == expected, f"{url} -> {wrapped._provider_name}, want {expected}"

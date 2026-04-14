"""Core instrumentation function."""

from __future__ import annotations

from typing import Optional, TypeVar
from urllib.parse import urlparse

from .transport import Transport

T = TypeVar("T")

# Default endpoint for the hosted platform
_DEFAULT_ENDPOINT = "https://spectracost.com/ingest"

# Maps known provider API hosts to the canonical provider name we store on
# events. An OpenAI() client pointing at one of these gets correctly
# attributed instead of being labelled "openai" by class.
_PROVIDER_BY_HOST: dict[str, str] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "google",
    "api.cohere.com": "cohere",
    "api.deepseek.com": "deepseek",
    "api.groq.com": "groq",
    "api.together.xyz": "together",
    "api.mistral.ai": "mistral",
    "api.x.ai": "xai",
    "openrouter.ai": "openrouter",
    "api.fireworks.ai": "fireworks",
}


def instrument(
    client: T,
    *,
    api_key: str,
    endpoint: str = _DEFAULT_ENDPOINT,
    provider: Optional[str] = None,
    team: str = "",
    service: str = "",
    feature: str = "",
    environment: str = "production",
    customer_id: str = "",
    tags: Optional[dict[str, str]] = None,
) -> T:
    """Wrap an AI provider client with Spectracost instrumentation.

    Returns a wrapped client that behaves identically to the original, but
    captures usage telemetry (model, tokens, latency, attribution tags) and
    sends it to the Spectracost platform asynchronously.

    Args:
        client: An OpenAI- or Anthropic-compatible client instance.
        api_key: Your Spectracost API key (starts with "sprc_").
        endpoint: Spectracost ingestion endpoint. Override for self-hosted or development.
        provider: Explicit provider name ("openai", "groq", "deepseek", etc.).
            If omitted, the provider is inferred from `client.base_url`: a client
            pointing at api.groq.com is attributed as groq even though it's an
            openai.OpenAI instance. Clients pointing at spectracost.com/proxy/v1/<p>/
            have their provider parsed from the path segment.
        team: Default team attribution tag.
        service: Default service attribution tag.
        feature: Default feature attribution tag.
        environment: Default environment ("production", "staging", "development", "test").
        customer_id: Default customer ID for unit economics tracking.
        tags: Additional custom tags to include on every event.

    Returns:
        A wrapped client with the same API as the original.

    Example:
        from openai import OpenAI
        from spectracost import instrument

        # OpenAI-compatible client pointing at Groq - auto-detected as groq
        client = instrument(
            OpenAI(base_url="https://api.groq.com/openai/v1"),
            api_key="sprc_...",
            team="search",
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    """
    transport = Transport(endpoint=endpoint, api_key=api_key)

    client_module = type(client).__module__
    client_name = type(client).__name__.lower()

    resolved_provider = _detect_provider(client, client_module, client_name, provider)

    wrapper_kwargs = dict(
        transport=transport,
        provider_name=resolved_provider,
        team=team,
        service=service,
        feature=feature,
        environment=environment,
        customer_id=customer_id,
        tags=tags,
    )

    if _is_openai(client, client_module, client_name):
        from .providers.openai import OpenAIWrapper
        return OpenAIWrapper(client, **wrapper_kwargs)  # type: ignore[return-value]

    if _is_anthropic(client, client_module, client_name):
        from .providers.anthropic import AnthropicWrapper
        return AnthropicWrapper(client, **wrapper_kwargs)  # type: ignore[return-value]

    raise ValueError(
        f"Unsupported client type: {type(client).__name__} (module: {client_module}). "
        f"Spectracost supports OpenAI and Anthropic clients."
    )


def _detect_provider(
    client: object,
    module: str,
    name: str,
    explicit: Optional[str],
) -> str:
    """Determine the logical provider name for an instrumented client.

    Priority:
      1. Explicit override via instrument(provider=...).
      2. client.base_url host: map to the canonical provider name.
      3. client.base_url path when pointed at Spectracost proxy.
      4. Duck-typed class check (openai.OpenAI -> "openai",
         anthropic.Anthropic -> "anthropic").
    """
    if explicit:
        return explicit

    base_url = getattr(client, "base_url", None)
    if base_url is not None:
        url_str = str(base_url)
        # MagicMocks stringify as "<MagicMock id='...' name='base_url'>"
        # in tests. Skip them so we fall through to the duck-type path.
        if not (url_str.startswith("<") and "Mock" in url_str):
            path_provider = _provider_from_spectracost_path(url_str)
            if path_provider:
                return path_provider
            host_provider = _provider_from_host(url_str)
            if host_provider:
                return host_provider

    # Class + duck-type fallback (matches the existing _is_* checks).
    if _is_anthropic(client, module, name):
        return "anthropic"
    if _is_openai(client, module, name):
        return "openai"
    return "openai"


def _provider_from_spectracost_path(url: str) -> Optional[str]:
    """Parse the provider segment out of a Spectracost proxy URL."""
    marker = "/proxy/v1/"
    if marker not in url:
        return None
    tail = url.split(marker, 1)[1]
    segments = [s for s in tail.split("/") if s]
    if not segments:
        return None
    candidate = segments[0]
    if candidate.startswith("sprc_"):
        return None
    return candidate or None


def _provider_from_host(url: str) -> Optional[str]:
    """Map a URL's host to a known provider name."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    host = (parsed.netloc or "").split(":", 1)[0].lower()
    if not host:
        return None
    for known, provider_name in _PROVIDER_BY_HOST.items():
        if host == known or host.endswith("." + known):
            return provider_name
    return None


def _is_openai(client: object, module: str, name: str) -> bool:
    """Detect OpenAI client by module, name, or duck typing."""
    if "openai" in module or "openai" in name:
        return True
    # Duck typing: OpenAI clients have client.chat.completions
    chat = getattr(client, "chat", None)
    return chat is not None and hasattr(chat, "completions")


def _is_anthropic(client: object, module: str, name: str) -> bool:
    """Detect Anthropic client by module, name, or duck typing."""
    if "anthropic" in module or "anthropic" in name:
        return True
    # Duck typing: Anthropic clients have client.messages but not client.chat
    return hasattr(client, "messages") and not hasattr(client, "chat")

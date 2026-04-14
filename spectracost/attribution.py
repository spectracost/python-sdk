"""Attribution context manager for per-call tag overrides."""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Generator


# Thread-safe and async-safe context variable
_attribution_ctx: ContextVar[dict[str, str]] = ContextVar("spectracost_attribution", default={})


@contextmanager
def attribution(
    *,
    team: Optional[str] = None,
    service: Optional[str] = None,
    feature: Optional[str] = None,
    environment: Optional[str] = None,
    customer_id: Optional[str] = None,
    **extra_tags: str,
) -> Generator[None, None, None]:
    """Override attribution tags for all instrumented calls within this scope.

    Tags set here are merged with client-level defaults, with context
    tags taking precedence.

    Usage:
        from spectracost import attribution

        with attribution(feature="semantic-search", customer_id="cust_abc"):
            response = client.chat.completions.create(...)
    """
    tags: dict[str, str] = {}
    if team is not None:
        tags["team"] = team
    if service is not None:
        tags["service"] = service
    if feature is not None:
        tags["feature"] = feature
    if environment is not None:
        tags["environment"] = environment
    if customer_id is not None:
        tags["customer_id"] = customer_id
    tags.update(extra_tags)

    # Merge with any existing parent context
    parent = _attribution_ctx.get()
    merged = {**parent, **tags}

    token = _attribution_ctx.set(merged)
    try:
        yield
    finally:
        _attribution_ctx.reset(token)


def get_current_attribution() -> dict[str, str]:
    """Get the current attribution context. Returns empty dict if no context is set."""
    return _attribution_ctx.get()

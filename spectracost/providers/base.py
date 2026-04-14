"""Base wrapper with shared interception logic."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..attribution import get_current_attribution
from ..transport import Transport
from ..types import UsageEvent

logger = logging.getLogger("spectracost")


class BaseWrapper:
    """Base class for provider-specific wrappers.

    Proxies all attribute access to the underlying client via __getattr__.
    Subclasses register interception hooks for specific API methods.
    """

    def __init__(
        self,
        client: Any,
        transport: Transport,
        *,
        provider_name: str,
        team: str = "",
        service: str = "",
        feature: str = "",
        environment: str = "production",
        customer_id: str = "",
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_transport", transport)
        object.__setattr__(self, "_provider_name", provider_name)
        object.__setattr__(self, "_defaults", {
            "team": team,
            "service": service,
            "feature": feature,
            "environment": environment,
            "customer_id": customer_id,
        })
        object.__setattr__(self, "_tags", tags or {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _get_attribution(self) -> dict[str, str]:
        """Merge client defaults with context overrides."""
        result = dict(self._defaults)
        ctx = get_current_attribution()
        result.update(ctx)
        return result

    def _build_event(
        self,
        *,
        provider: str,
        model: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        status: str = "success",
        error_code: str = "",
        cached_tokens: int = 0,
        time_to_first_token_ms: int = 0,
        provider_request_id: str = "",
        prompt_hash: str = "",
    ) -> UsageEvent:
        """Construct a UsageEvent with current attribution context."""
        attrs = self._get_attribution()
        return UsageEvent(
            provider=provider,
            model=model,
            endpoint=endpoint,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            status=status,
            error_code=error_code,
            cached_tokens=cached_tokens,
            time_to_first_token_ms=time_to_first_token_ms,
            team=attrs.get("team", ""),
            service=attrs.get("service", ""),
            feature=attrs.get("feature", ""),
            environment=attrs.get("environment", "production"),
            customer_id=attrs.get("customer_id", ""),
            tags=self._tags,
            provider_request_id=provider_request_id,
            prompt_hash=prompt_hash,
        )

    def _emit(self, event: UsageEvent) -> None:
        """Send event to transport. Never raises."""
        try:
            self._transport.enqueue(event)
        except Exception:
            logger.debug("spectracost: failed to enqueue event", exc_info=True)

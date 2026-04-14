"""Event types for Spectracost telemetry."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict


@dataclass
class UsageEvent:
    """A single AI API call usage event."""

    provider: str
    model: str
    endpoint: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    status: str  # "success", "error", "rate_limited", "timeout"

    # Cost — filled by the platform, not the SDK
    input_cost_usd: str = "0"
    output_cost_usd: str = "0"
    total_cost_usd: str = "0"

    # Attribution
    team: str = ""
    service: str = ""
    feature: str = ""
    environment: str = "production"
    customer_id: str = ""

    # Optional fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()))
    cached_tokens: int = 0
    time_to_first_token_ms: int = 0
    error_code: str = ""
    model_version: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    idempotency_key: str = ""
    provider_request_id: str = ""
    prompt_hash: str = ""
    prompt_template_id: str = ""

    def to_dict(self) -> dict:
        """Serialize to a dict suitable for JSON encoding."""
        return {k: v for k, v in asdict(self).items() if v or isinstance(v, int)}

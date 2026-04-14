"""Anthropic provider wrapper."""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import BaseWrapper

logger = logging.getLogger("spectracost")


class _MessagesWrapper:
    """Wraps client.messages to intercept create() and stream()."""

    def __init__(self, messages: Any, wrapper: "AnthropicWrapper") -> None:
        self._messages = messages
        self._wrapper = wrapper

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        stream = kwargs.get("stream", False)
        start_ms = time.monotonic_ns() // 1_000_000

        try:
            response = self._messages.create(**kwargs)
        except Exception as exc:
            latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
            self._wrapper._emit(self._wrapper._build_event(
                provider=self._wrapper._provider_name,
                model=model,
                endpoint="messages",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                status="error",
                error_code=type(exc).__name__,
            ))
            raise

        if stream:
            return self._wrap_stream(response, model, start_ms)

        latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
        try:
            usage = response.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            cached = 0
            if hasattr(usage, "cache_read_input_tokens"):
                cached = usage.cache_read_input_tokens or 0
        except Exception:
            input_tokens = 0
            output_tokens = 0
            cached = 0

        self._wrapper._emit(self._wrapper._build_event(
            provider=self._wrapper._provider_name,
            model=model,
            endpoint="messages",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cached_tokens=cached,
        ))

        return response

    def _wrap_stream(self, stream: Any, model: str, start_ms: int) -> Any:
        """Wrap a streaming response, emitting telemetry after stream completes."""
        input_tokens = 0
        output_tokens = 0
        first_token_ms = 0
        got_first = False

        try:
            for event in stream:
                if not got_first:
                    first_token_ms = (time.monotonic_ns() // 1_000_000) - start_ms
                    got_first = True

                # Anthropic streams emit a message_delta event with usage at the end
                if hasattr(event, "type"):
                    if event.type == "message_start" and hasattr(event, "message"):
                        msg_usage = getattr(event.message, "usage", None)
                        if msg_usage:
                            input_tokens = getattr(msg_usage, "input_tokens", 0) or 0
                    elif event.type == "message_delta":
                        delta_usage = getattr(event, "usage", None)
                        if delta_usage:
                            output_tokens = getattr(delta_usage, "output_tokens", 0) or 0

                yield event
        except Exception as exc:
            latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
            self._wrapper._emit(self._wrapper._build_event(
                provider=self._wrapper._provider_name,
                model=model,
                endpoint="messages",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                status="error",
                error_code=type(exc).__name__,
                time_to_first_token_ms=first_token_ms,
            ))
            raise

        latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
        self._wrapper._emit(self._wrapper._build_event(
            provider=self._wrapper._provider_name,
            model=model,
            endpoint="messages",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            time_to_first_token_ms=first_token_ms,
        ))


class AnthropicWrapper(BaseWrapper):
    """Wraps an Anthropic client to capture usage telemetry."""

    def __init__(self, client: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(client, *args, **kwargs)
        object.__setattr__(self, "messages", _MessagesWrapper(client.messages, self))

"""OpenAI provider wrapper."""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import BaseWrapper

logger = logging.getLogger("spectracost")


class _ChatCompletionsWrapper:
    """Wraps client.chat.completions to intercept create()."""

    def __init__(self, completions: Any, wrapper: "OpenAIWrapper") -> None:
        self._completions = completions
        self._wrapper = wrapper

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        stream = kwargs.get("stream", False)
        start_ms = time.monotonic_ns() // 1_000_000

        try:
            response = self._completions.create(**kwargs)
        except Exception as exc:
            latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
            self._wrapper._emit(self._wrapper._build_event(
                provider=self._wrapper._provider_name,
                model=model,
                endpoint="chat.completions",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                status="error",
                error_code=type(exc).__name__,
            ))
            raise

        if stream:
            return self._wrap_stream(response, model, start_ms)

        # Non-streaming: extract usage from response
        latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
        try:
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cached_tokens = getattr(usage, "prompt_tokens_details", None)
            cached = 0
            if cached_tokens and hasattr(cached_tokens, "cached_tokens"):
                cached = cached_tokens.cached_tokens or 0
        except Exception:
            input_tokens = 0
            output_tokens = 0
            cached = 0

        self._wrapper._emit(self._wrapper._build_event(
            provider=self._wrapper._provider_name,
            model=model,
            endpoint="chat.completions",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cached_tokens=cached,
        ))

        return response

    def _wrap_stream(self, stream: Any, model: str, start_ms: int) -> Any:
        """Wrap a streaming response, emitting telemetry after the stream completes."""
        input_tokens = 0
        output_tokens = 0
        first_token_ms = 0
        got_first = False

        try:
            for chunk in stream:
                if not got_first:
                    first_token_ms = (time.monotonic_ns() // 1_000_000) - start_ms
                    got_first = True

                # Extract usage from final chunk if available
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

                yield chunk
        except Exception as exc:
            latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
            self._wrapper._emit(self._wrapper._build_event(
                provider=self._wrapper._provider_name,
                model=model,
                endpoint="chat.completions",
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
            endpoint="chat.completions",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            time_to_first_token_ms=first_token_ms,
        ))


class _ChatWrapper:
    """Wraps client.chat to provide wrapped completions."""

    def __init__(self, chat: Any, wrapper: "OpenAIWrapper") -> None:
        self._chat = chat
        self._wrapper = wrapper
        self.completions = _ChatCompletionsWrapper(chat.completions, wrapper)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _EmbeddingsWrapper:
    """Wraps client.embeddings to intercept create()."""

    def __init__(self, embeddings: Any, wrapper: "OpenAIWrapper") -> None:
        self._embeddings = embeddings
        self._wrapper = wrapper

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        start_ms = time.monotonic_ns() // 1_000_000

        try:
            response = self._embeddings.create(**kwargs)
        except Exception as exc:
            latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
            self._wrapper._emit(self._wrapper._build_event(
                provider=self._wrapper._provider_name,
                model=model,
                endpoint="embeddings",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                status="error",
                error_code=type(exc).__name__,
            ))
            raise

        latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
        try:
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
        except Exception:
            input_tokens = 0

        self._wrapper._emit(self._wrapper._build_event(
            provider=self._wrapper._provider_name,
            model=model,
            endpoint="embeddings",
            input_tokens=input_tokens,
            output_tokens=0,
            latency_ms=latency_ms,
        ))

        return response


class OpenAIWrapper(BaseWrapper):
    """Wraps an OpenAI client to capture usage telemetry."""

    def __init__(self, client: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(client, *args, **kwargs)
        object.__setattr__(self, "chat", _ChatWrapper(client.chat, self))
        object.__setattr__(self, "embeddings", _EmbeddingsWrapper(client.embeddings, self))

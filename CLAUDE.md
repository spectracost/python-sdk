# Python SDK - Agent Context

## What This Is

The Spectracost Python SDK is an open-source (MIT) library that wraps AI provider clients (OpenAI, Anthropic) to capture usage telemetry. It is the primary customer-facing component of the platform. More customers will interact with this code than any other part of the system.

## Critical Constraints

1. **Zero required dependencies beyond the provider SDK.** A customer who uses `openai` should be able to `pip install spectracost` without pulling in a dependency tree. The only imports at call time should be the provider's SDK and ours. Use only stdlib for the transport layer (urllib, threading, json, logging).

2. **Never break the customer's LLM calls.** Every exception in our instrumentation code must be caught and silently logged. The wrapped client must behave identically to the unwrapped client in every case - same return values, same exceptions, same streaming behavior. If our code raises, we've failed.

3. **<5ms latency overhead.** The instrumentation adds a before-hook (capture start time, extract tags) and an after-hook (extract token counts, queue telemetry event). Both must be fast. Telemetry emission is async - it never blocks the return of the LLM response.

4. **This is a public API.** Every public function, class, and method must have type hints and a docstring. Breaking changes require a semver major version bump. Design the API surface carefully - every function we expose is a commitment.

## Architecture

```
spectracost/
├── __init__.py          # Public API: exports `instrument`, `attribution`
├── instrument.py        # Core: `instrument(client, ...)` wrapper function
├── attribution.py       # Context manager for per-call attribution overrides
├── transport.py         # Background thread that batches and sends events
├── types.py             # UsageEvent dataclass, type definitions
├── pricing.py           # Local pricing table for standalone mode
└── providers/
    ├── __init__.py
    ├── base.py          # Base wrapper class with shared interception logic
    ├── openai.py        # OpenAI-specific wrapper (chat.completions, embeddings)
    └── anthropic.py     # Anthropic-specific wrapper (messages)
```

### How Wrapping Works

`instrument()` inspects the client type, selects the appropriate provider wrapper, and returns a proxy object that:

1. Delegates all attribute access to the underlying client via `__getattr__`
2. Intercepts known API methods (e.g., `chat.completions.create`) with a before/after hook
3. The before hook captures: start time, model, attribution tags from context
4. The after hook captures: end time, token counts (from response), status
5. Constructs a `UsageEvent` and queues it in the transport buffer
6. Returns the original response unmodified

For **streaming responses** (SSE):
- The wrapper returns a generator/async generator that passes through chunks from the provider
- Token counts are extracted from the final chunk (providers include `usage` in the last SSE event)
- The telemetry event is emitted after the stream completes, not before

### Transport Layer

The `Transport` class runs a background `threading.Thread` that:
- Dequeues events from an in-memory `queue.Queue`
- Batches events (up to 100 events or 1 second, whichever comes first)
- POSTs the batch to `{endpoint}/v1/events` as JSON
- On HTTP failure: holds events in buffer, retries with exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Buffer cap: 10MB. If exceeded, oldest events are dropped with a warning log.
- On interpreter shutdown (`atexit`): flushes remaining buffer with a 5-second timeout

Uses `threading.Thread`, NOT `asyncio`, because the SDK must work in non-async codebases. The thread is a daemon thread so it doesn't prevent process exit.

### Attribution Context

```python
from spectracost import attribution

with attribution(feature="semantic-search", customer_id="cust_abc"):
    response = client.chat.completions.create(...)
```

Uses `contextvars.ContextVar` for thread-safe, async-safe context propagation. Tags set in the context manager override the client-level defaults for calls within that scope. Tags are merged: client defaults + context overrides, with context taking precedence.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

### Test Categories

1. **Wrapper transparency tests**: verify that wrapping a client doesn't change its behavior. Mock the provider client, wrap it, call methods, assert same return values and same exceptions.

2. **Telemetry correctness tests**: verify that the right events are generated with the right fields. Use a mock HTTP server (pytest-httpserver) as the telemetry endpoint.

3. **Failure isolation tests**: verify that telemetry failures don't affect LLM calls. Point the transport at a dead endpoint, make calls, assert they still return correctly.

4. **Streaming tests**: verify that streaming responses are passed through correctly and token counts are captured from the final chunk.

5. **Attribution context tests**: verify that context manager tags override client defaults and that nested contexts work correctly.

### What NOT to Test

- Don't test the actual OpenAI/Anthropic APIs (they require real API keys and cost money)
- Don't test internal implementation details of the transport batching algorithm
- Don't test that Python's `threading.Thread` works correctly

## Common Pitfalls

- **Don't import provider SDKs at module level.** Use conditional imports. A customer who only uses OpenAI shouldn't need Anthropic installed, and vice versa. The provider-specific wrapper modules should handle ImportError gracefully.
- **Don't mutate the original client.** The wrapper proxies to the original client - it never modifies it. The customer should be able to use both the original and wrapped client interchangeably.
- **Don't assume response structure.** Provider SDKs evolve. Access response fields defensively. If token counts aren't where we expect, log a warning and emit the event with zero tokens - don't crash.
- **Don't use global state for transport.** Each `instrument()` call creates its own transport instance. Multiple wrapped clients can coexist with different endpoints and API keys.

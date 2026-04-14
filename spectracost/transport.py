"""Async transport layer — batches and sends telemetry events in a background thread."""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
import urllib.request
import urllib.error
import weakref

from .types import UsageEvent

logger = logging.getLogger("spectracost")

_MAX_BUFFER_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_BATCH_SIZE = 100
_FLUSH_INTERVAL_SECONDS = 1.0
_MAX_RETRY_DELAY = 60.0
_SHUTDOWN_TIMEOUT = 5.0

# Weak references to every Transport instance ever created. Used by
# tests to shut down leaked transports between cases so one test's
# retrying background thread can't leak POSTs into the next test's
# fixture server.
_ALL_TRANSPORTS: "weakref.WeakSet[Transport]" = weakref.WeakSet()


class Transport:
    """Background transport that batches events and POSTs them to the ingestion endpoint."""

    def __init__(self, endpoint: str, api_key: str) -> None:
        self._endpoint = endpoint.rstrip("/") + "/v1/events"
        self._api_key = api_key
        self._queue: queue.Queue[UsageEvent] = queue.Queue()
        self._buffer_bytes = 0
        self._shutdown = threading.Event()
        self._retry_delay = 1.0

        self._thread = threading.Thread(target=self._run, daemon=True, name="spectracost-transport")
        self._thread.start()

        _ALL_TRANSPORTS.add(self)
        atexit.register(self.flush)

    def enqueue(self, event: UsageEvent) -> None:
        """Add an event to the send queue. Drops oldest events if buffer is full."""
        event_size = len(json.dumps(event.to_dict()).encode())

        if self._buffer_bytes + event_size > _MAX_BUFFER_BYTES:
            logger.warning("spectracost: buffer full (%d bytes), dropping oldest event", self._buffer_bytes)
            try:
                dropped = self._queue.get_nowait()
                dropped_size = len(json.dumps(dropped.to_dict()).encode())
                self._buffer_bytes -= dropped_size
            except queue.Empty:
                pass

        self._queue.put(event)
        self._buffer_bytes += event_size

    def flush(self) -> None:
        """Flush all pending events. Called on shutdown."""
        self._shutdown.set()
        self._thread.join(timeout=_SHUTDOWN_TIMEOUT)

    def _run(self) -> None:
        """Background thread main loop."""
        batch: list[UsageEvent] = []
        last_flush = time.monotonic()

        while True:
            try:
                timeout = max(0.01, _FLUSH_INTERVAL_SECONDS - (time.monotonic() - last_flush))
                event = self._queue.get(timeout=timeout)
                batch.append(event)
            except queue.Empty:
                pass

            should_flush = (
                len(batch) >= _MAX_BATCH_SIZE
                or (batch and time.monotonic() - last_flush >= _FLUSH_INTERVAL_SECONDS)
                or (self._shutdown.is_set() and batch)
            )

            if should_flush:
                self._send_batch(batch)
                for evt in batch:
                    evt_size = len(json.dumps(evt.to_dict()).encode())
                    self._buffer_bytes = max(0, self._buffer_bytes - evt_size)
                batch = []
                last_flush = time.monotonic()

            if self._shutdown.is_set() and self._queue.empty() and not batch:
                break

    def _send_batch(self, batch: list[UsageEvent]) -> None:
        """Send a batch of events to the ingestion endpoint."""
        if not batch:
            return

        payload = json.dumps([e.to_dict() for e in batch]).encode("utf-8")

        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "spectracost-sdk/0.1.1",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            self._retry_delay = 1.0  # reset on success
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            logger.debug("spectracost: failed to send batch (%d events): %s", len(batch), exc)
            # Re-enqueue events for retry
            for event in batch:
                self._queue.put(event)
            time.sleep(min(self._retry_delay, _MAX_RETRY_DELAY))
            self._retry_delay = min(self._retry_delay * 2, _MAX_RETRY_DELAY)


def _shutdown_all_active_transports() -> None:
    """Shut down every live Transport. Used by tests to guarantee isolation.

    Not part of the public API. The critical step is setting _shutdown
    and draining the queue so no further POSTs go out; the background
    thread is a daemon and will exit on its own if it lingers.
    """
    for t in list(_ALL_TRANSPORTS):
        t._shutdown.set()
        while True:
            try:
                t._queue.get_nowait()
            except queue.Empty:
                break
    _ALL_TRANSPORTS.clear()

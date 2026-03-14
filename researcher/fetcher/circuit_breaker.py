"""
Per-host circuit breaker.

States:
  CLOSED   — normal; requests pass through, failures are counted.
  OPEN     — circuit tripped; requests are rejected immediately.
  (auto)   — after the pause expires, the next is_open() call resets the host
              back to CLOSED so traffic can resume.

Transitions:
  threshold consecutive 429/5xx  → CLOSED → OPEN (blocked for pause_minutes)
  pause expires                   → OPEN   → CLOSED (automatic)
  record_success()                → any    → CLOSED (reset counter)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class _HostState:
    failures: int = 0
    open_until: float = 0.0  # monotonic clock timestamp; 0 = not open


class CircuitBreaker:
    """Async-safe (single-threaded event loop) per-host circuit breaker."""

    def __init__(self, threshold: int = 5, pause_minutes: float = 10.0):
        self._threshold = threshold
        self._pause_seconds = pause_minutes * 60.0
        self._hosts: dict[str, _HostState] = {}

    # ── internal ──────────────────────────────────────────────────────────

    def _state(self, host: str) -> _HostState:
        if host not in self._hosts:
            self._hosts[host] = _HostState()
        return self._hosts[host]

    # ── public API ────────────────────────────────────────────────────────

    def is_open(self, host: str) -> bool:
        """Return True if the circuit is open and requests should be blocked."""
        s = self._state(host)
        if s.open_until == 0.0:
            return False
        if time.monotonic() >= s.open_until:
            # Pause expired — reset and allow traffic again
            s.failures = 0
            s.open_until = 0.0
            log.info("CircuitBreaker: circuit closed for %s (pause expired)", host)
            return False
        return True

    def record_success(self, host: str) -> None:
        """Call after any successful response. Resets the failure counter."""
        s = self._state(host)
        if s.failures > 0 or s.open_until > 0.0:
            log.debug(
                "CircuitBreaker: success — resetting %s (was %d failures)", host, s.failures
            )
        s.failures = 0
        s.open_until = 0.0

    def record_failure(self, host: str) -> None:
        """Call on 429 or 5xx responses. May trip the circuit."""
        s = self._state(host)
        s.failures += 1
        if s.failures >= self._threshold:
            s.open_until = time.monotonic() + self._pause_seconds
            log.warning(
                "CircuitBreaker: circuit OPENED for %s after %d failures "
                "(paused %.0f min)",
                host, s.failures, self._pause_seconds / 60.0,
            )

    def failure_count(self, host: str) -> int:
        return self._state(host).failures

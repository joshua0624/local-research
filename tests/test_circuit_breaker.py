"""Tests for the per-host circuit breaker."""
from __future__ import annotations

import time
import unittest.mock

import pytest

from researcher.fetcher.circuit_breaker import CircuitBreaker


def make_cb(threshold: int = 3, pause_minutes: float = 10.0) -> CircuitBreaker:
    return CircuitBreaker(threshold=threshold, pause_minutes=pause_minutes)


class TestCircuitBreakerInitial:
    def test_new_breaker_is_closed(self):
        cb = make_cb()
        assert cb.is_open("example.com") is False

    def test_unknown_host_is_closed(self):
        cb = make_cb()
        assert cb.is_open("never-seen.com") is False

    def test_failure_count_starts_at_zero(self):
        cb = make_cb()
        assert cb.failure_count("example.com") == 0


class TestFailureAccumulation:
    def test_failures_below_threshold_keep_circuit_closed(self):
        cb = make_cb(threshold=3)
        cb.record_failure("api.example.com")
        cb.record_failure("api.example.com")
        assert cb.is_open("api.example.com") is False
        assert cb.failure_count("api.example.com") == 2

    def test_failures_at_threshold_open_circuit(self):
        cb = make_cb(threshold=3)
        for _ in range(3):
            cb.record_failure("api.example.com")
        assert cb.is_open("api.example.com") is True

    def test_failures_beyond_threshold_stay_open(self):
        cb = make_cb(threshold=3)
        for _ in range(10):
            cb.record_failure("api.example.com")
        assert cb.is_open("api.example.com") is True

    def test_failures_are_per_host(self):
        cb = make_cb(threshold=2)
        cb.record_failure("host-a.com")
        cb.record_failure("host-a.com")  # opens host-a
        cb.record_failure("host-b.com")  # host-b: only 1 failure
        assert cb.is_open("host-a.com") is True
        assert cb.is_open("host-b.com") is False


class TestSuccessReset:
    def test_success_resets_failure_count(self):
        cb = make_cb(threshold=3)
        cb.record_failure("api.example.com")
        cb.record_failure("api.example.com")
        cb.record_success("api.example.com")
        assert cb.failure_count("api.example.com") == 0
        assert cb.is_open("api.example.com") is False

    def test_success_closes_open_circuit(self):
        cb = make_cb(threshold=2)
        cb.record_failure("api.example.com")
        cb.record_failure("api.example.com")
        assert cb.is_open("api.example.com") is True
        cb.record_success("api.example.com")
        assert cb.is_open("api.example.com") is False

    def test_success_on_unknown_host_is_safe(self):
        cb = make_cb()
        cb.record_success("new-host.com")  # should not raise
        assert cb.failure_count("new-host.com") == 0


class TestPauseExpiry:
    def test_circuit_auto_resets_after_pause(self):
        cb = make_cb(threshold=1, pause_minutes=1.0 / 60.0)  # 1-second pause
        cb.record_failure("api.example.com")
        assert cb.is_open("api.example.com") is True

        # Mock time to be past the pause
        future = time.monotonic() + 2.0
        with unittest.mock.patch("time.monotonic", return_value=future):
            assert cb.is_open("api.example.com") is False

    def test_circuit_remains_open_before_pause_expires(self):
        cb = make_cb(threshold=1, pause_minutes=10.0)
        cb.record_failure("api.example.com")
        # Mock time to be before pause expires
        soon = time.monotonic() + 60.0  # only 1 minute later, pause is 10 min
        with unittest.mock.patch("time.monotonic", return_value=soon):
            assert cb.is_open("api.example.com") is True

    def test_after_auto_reset_failures_start_fresh(self):
        cb = make_cb(threshold=1, pause_minutes=1.0 / 60.0)
        cb.record_failure("api.example.com")

        future = time.monotonic() + 2.0
        with unittest.mock.patch("time.monotonic", return_value=future):
            # Auto-reset
            assert cb.is_open("api.example.com") is False
            assert cb.failure_count("api.example.com") == 0

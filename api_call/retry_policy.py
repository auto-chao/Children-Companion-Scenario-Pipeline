"""Shared retry helpers for HTTP / OpenAI-compatible API calls (rate limit, timeouts, etc.)."""

from __future__ import annotations

import time


def is_retryable_error_message(msg: str) -> bool:
    """True if the error string suggests a transient failure worth retrying."""
    m = msg.lower()
    return (
        "429" in m
        or "resource exhausted" in m
        or "quota" in m
        or "rate" in m
        or "503" in m
        or "502" in m
        or "500" in m
        or "timeout" in m
        or "timed out" in m
        or "connection" in m
        or "connect" in m
        or "network" in m
        or "broken pipe" in m
        or "reset" in m
        or "stream" in m
    )


def sleep_before_next_attempt(current_sleep_s: float, *, cap_s: float = 60.0) -> float:
    """
    Block with exponential backoff step, return the sleep duration to use before the *following* retry.

    Matches existing pipeline scripts: sleep, then double until cap.
    """
    time.sleep(current_sleep_s)
    return min(current_sleep_s * 2, cap_s)

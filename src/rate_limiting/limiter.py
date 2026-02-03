"""Rate limiting implementation using token bucket algorithm."""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 3000
    requests_per_day: int = 50000
    tokens_per_minute: int = 100000
    concurrent_requests: int = 10


class TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None):
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
            timeout: Maximum wait time in seconds

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        start_time = time.time()

        while True:
            if await self.consume(tokens):
                return

            # Calculate wait time
            async with self._lock:
                self._refill()
                needed = tokens - self.tokens
                wait_time = needed / self.refill_rate if self.refill_rate > 0 else 1

            wait_time = min(wait_time, 1.0)  # Max 1 second wait

            if timeout and (time.time() - start_time + wait_time) > timeout:
                raise asyncio.TimeoutError("Rate limit wait timeout exceeded")

            await asyncio.sleep(wait_time)


class RateLimiter:
    """
    Multi-tier rate limiter for LLM API calls.

    Implements rate limiting at multiple time scales (minute, hour, day)
    and tracks both request counts and token usage.
    """

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with configuration."""
        self.config = config

        # Request rate limiters
        self.minute_limiter = TokenBucket(
            capacity=config.requests_per_minute,
            refill_rate=config.requests_per_minute / 60.0
        )
        self.hour_limiter = TokenBucket(
            capacity=config.requests_per_hour,
            refill_rate=config.requests_per_hour / 3600.0
        )
        self.day_limiter = TokenBucket(
            capacity=config.requests_per_day,
            refill_rate=config.requests_per_day / 86400.0
        )

        # Token usage limiter
        self.token_limiter = TokenBucket(
            capacity=config.tokens_per_minute,
            refill_rate=config.tokens_per_minute / 60.0
        )

        # Concurrent request semaphore
        self.concurrent_semaphore = asyncio.Semaphore(config.concurrent_requests)

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_time = 0.0

    async def acquire(self, estimated_tokens: int = 1000, timeout: Optional[float] = None):
        """
        Acquire permission to make an API call.

        Args:
            estimated_tokens: Estimated tokens for the request
            timeout: Maximum wait time in seconds

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        start_time = time.time()

        # Wait for concurrent request slot
        await asyncio.wait_for(
            self.concurrent_semaphore.acquire(),
            timeout=timeout
        )

        try:
            # Wait for rate limit tokens
            remaining_timeout = None
            if timeout:
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout <= 0:
                    raise asyncio.TimeoutError("Rate limit timeout exceeded")

            # Check all rate limiters
            await self.minute_limiter.wait_for_tokens(1, remaining_timeout)
            await self.hour_limiter.wait_for_tokens(1, remaining_timeout)
            await self.day_limiter.wait_for_tokens(1, remaining_timeout)
            await self.token_limiter.wait_for_tokens(estimated_tokens, remaining_timeout)

            # Update statistics
            self.total_requests += 1
            self.total_tokens += estimated_tokens
            self.total_wait_time += time.time() - start_time

        except Exception:
            # Release semaphore on error
            self.concurrent_semaphore.release()
            raise

    def release(self):
        """Release concurrent request slot."""
        self.concurrent_semaphore.release()

    def get_statistics(self) -> Dict[str, any]:
        """Get rate limiter statistics."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_wait_time": self.total_wait_time,
            "avg_wait_time": self.total_wait_time / max(self.total_requests, 1),
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
                "tokens_per_minute": self.config.tokens_per_minute,
                "concurrent_requests": self.config.concurrent_requests,
            }
        }

"""Failure handling and retry logic."""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from ..agents.base import Agent, AgentResult, AgentStatus


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CONSTANT_DELAY = "constant_delay"
    IMMEDIATE = "immediate"


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.strategy == RetryStrategy.IMMEDIATE:
            return 0.0

        elif self.strategy == RetryStrategy.CONSTANT_DELAY:
            delay = self.base_delay

        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt

        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        else:
            delay = self.base_delay

        # Apply max delay
        delay = min(delay, self.max_delay)

        # Apply jitter
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)

        return delay


class FailureHandler:
    """
    Handles agent failures and implements retry logic.
    """

    def __init__(
        self,
        default_policy: Optional[RetryPolicy] = None,
        on_failure: Optional[Callable] = None
    ):
        """
        Initialize failure handler.

        Args:
            default_policy: Default retry policy
            on_failure: Callback for failures
        """
        self.default_policy = default_policy or RetryPolicy()
        self.on_failure = on_failure

        # Statistics
        self.total_failures = 0
        self.total_retries = 0
        self.permanent_failures = 0

    async def handle_failure(
        self,
        agent: Agent,
        result: AgentResult,
        policy: Optional[RetryPolicy] = None
    ) -> bool:
        """
        Handle agent failure and determine if retry should occur.

        Args:
            agent: Failed agent
            result: Agent result with failure information
            policy: Retry policy (uses default if not provided)

        Returns:
            True if agent should be retried, False otherwise
        """
        self.total_failures += 1

        # Call failure callback
        if self.on_failure:
            try:
                await self.on_failure(agent, result)
            except Exception as e:
                print(f"Error in failure callback: {e}")

        # Check if retry is possible
        if not result.is_retriable():
            self.permanent_failures += 1
            return False

        # Check retry count
        retry_policy = policy or self.default_policy
        if result.retry_count >= retry_policy.max_retries:
            self.permanent_failures += 1
            return False

        # Calculate delay and wait
        delay = retry_policy.get_delay(result.retry_count + 1)
        if delay > 0:
            await asyncio.sleep(delay)

        self.total_retries += 1
        return True

    def should_fail_workflow(
        self,
        failed_agents: list,
        total_agents: int,
        allow_partial_completion: bool
    ) -> bool:
        """
        Determine if workflow should fail based on agent failures.

        Args:
            failed_agents: List of failed agents
            total_agents: Total number of agents
            allow_partial_completion: Whether partial completion is allowed

        Returns:
            True if workflow should fail, False if it can continue
        """
        if not failed_agents:
            return False

        if not allow_partial_completion:
            return True

        # Check if critical agents failed
        critical_failures = [
            agent for agent in failed_agents
            if agent.priority >= 8  # High priority agents are critical
        ]

        if critical_failures:
            return True

        # Check failure rate
        failure_rate = len(failed_agents) / total_agents
        if failure_rate > 0.5:  # More than 50% failed
            return True

        return False

    def get_statistics(self) -> dict:
        """Get failure handling statistics."""
        return {
            "total_failures": self.total_failures,
            "total_retries": self.total_retries,
            "permanent_failures": self.permanent_failures,
            "retry_rate": self.total_retries / max(self.total_failures, 1),
        }

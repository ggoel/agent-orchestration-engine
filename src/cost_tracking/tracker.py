"""Cost tracking and budget management."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class CostConfig:
    """Cost configuration for different models."""
    model_costs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    })
    default_input_cost: float = 0.01
    default_output_cost: float = 0.03


@dataclass
class CostEntry:
    """Single cost entry."""
    timestamp: datetime
    agent_id: str
    agent_name: str
    workflow_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    metadata: Dict = field(default_factory=dict)


class CostTracker:
    """
    Tracks costs across all agent executions and enforces budgets.
    """

    def __init__(self, config: CostConfig):
        """Initialize cost tracker."""
        self.config = config
        self._lock = asyncio.Lock()

        # Cost entries
        self.entries: List[CostEntry] = []

        # Budget tracking
        self.workflow_budgets: Dict[str, float] = {}
        self.workflow_costs: Dict[str, float] = {}
        self.agent_budgets: Dict[str, float] = {}
        self.agent_costs: Dict[str, float] = {}

        # Totals
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a model invocation.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in dollars
        """
        if model in self.config.model_costs:
            costs = self.config.model_costs[model]
            input_cost = costs["input"]
            output_cost = costs["output"]
        else:
            input_cost = self.config.default_input_cost
            output_cost = self.config.default_output_cost

        # Costs are per 1000 tokens
        total_cost = (
            (input_tokens / 1000.0) * input_cost +
            (output_tokens / 1000.0) * output_cost
        )

        return total_cost

    async def record_cost(
        self,
        agent_id: str,
        agent_name: str,
        workflow_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Record cost for an agent execution.

        Returns:
            Cost for this execution
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        entry = CostEntry(
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            agent_name=agent_name,
            workflow_id=workflow_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            metadata=metadata or {}
        )

        async with self._lock:
            self.entries.append(entry)

            # Update totals
            self.total_cost += cost
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            # Update workflow costs
            self.workflow_costs[workflow_id] = (
                self.workflow_costs.get(workflow_id, 0.0) + cost
            )

            # Update agent costs
            self.agent_costs[agent_id] = (
                self.agent_costs.get(agent_id, 0.0) + cost
            )

        return cost

    async def set_workflow_budget(self, workflow_id: str, budget: float):
        """Set budget for a workflow."""
        async with self._lock:
            self.workflow_budgets[workflow_id] = budget

    async def set_agent_budget(self, agent_id: str, budget: float):
        """Set budget for an agent."""
        async with self._lock:
            self.agent_budgets[agent_id] = budget

    async def check_workflow_budget(self, workflow_id: str) -> bool:
        """
        Check if workflow is within budget.

        Returns:
            True if within budget, False if exceeded
        """
        async with self._lock:
            if workflow_id not in self.workflow_budgets:
                return True

            budget = self.workflow_budgets[workflow_id]
            spent = self.workflow_costs.get(workflow_id, 0.0)
            return spent < budget

    async def check_agent_budget(self, agent_id: str) -> bool:
        """
        Check if agent is within budget.

        Returns:
            True if within budget, False if exceeded
        """
        async with self._lock:
            if agent_id not in self.agent_budgets:
                return True

            budget = self.agent_budgets[agent_id]
            spent = self.agent_costs.get(agent_id, 0.0)
            return spent < budget

    async def get_workflow_cost(self, workflow_id: str) -> float:
        """Get total cost for a workflow."""
        async with self._lock:
            return self.workflow_costs.get(workflow_id, 0.0)

    async def get_agent_cost(self, agent_id: str) -> float:
        """Get total cost for an agent."""
        async with self._lock:
            return self.agent_costs.get(agent_id, 0.0)

    def get_statistics(self) -> Dict:
        """Get cost statistics."""
        workflow_stats = []
        for workflow_id, spent in self.workflow_costs.items():
            budget = self.workflow_budgets.get(workflow_id)
            workflow_stats.append({
                "workflow_id": workflow_id,
                "spent": spent,
                "budget": budget,
                "remaining": budget - spent if budget else None,
                "utilization": (spent / budget * 100) if budget else None
            })

        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_entries": len(self.entries),
            "workflow_stats": workflow_stats,
        }

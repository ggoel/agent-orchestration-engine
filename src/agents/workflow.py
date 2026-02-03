"""Workflow management for agent orchestration."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import Agent, AgentStatus


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Workflow:
    """
    A workflow represents a collection of agents with dependencies.

    Attributes:
        name: Workflow name
        description: Workflow purpose
        agents: List of agents in the workflow
        max_parallel_agents: Maximum agents to run in parallel
        cost_budget: Maximum cost for entire workflow
        timeout: Workflow timeout in seconds
        allow_partial_completion: Continue execution on non-critical failures
    """
    name: str
    description: str
    agents: List[Agent]
    max_parallel_agents: int = 10
    cost_budget: float = 100.0
    timeout: float = 3600.0
    allow_partial_completion: bool = True

    # Runtime fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_cost: float = 0.0
    total_tokens: int = 0

    # Metadata
    module: str = "General"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate workflow configuration."""
        if not self.agents:
            raise ValueError("Workflow must have at least one agent")
        if self.max_parallel_agents < 1:
            raise ValueError("max_parallel_agents must be >= 1")
        if self.cost_budget <= 0:
            raise ValueError("cost_budget must be > 0")

        # Validate no circular dependencies
        self._validate_dependencies()

    def _validate_dependencies(self):
        """Validate workflow has no circular dependencies."""
        agent_names = {agent.name for agent in self.agents}

        # Check all dependencies exist
        for agent in self.agents:
            for dep in agent.dependencies:
                if dep not in agent_names:
                    raise ValueError(
                        f"Agent {agent.name} depends on non-existent agent {dep}"
                    )

        # Check for circular dependencies using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(agent_name: str) -> bool:
            visited.add(agent_name)
            rec_stack.add(agent_name)

            agent = next(a for a in self.agents if a.name == agent_name)
            for dep in agent.dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(agent_name)
            return False

        for agent in self.agents:
            if agent.name not in visited:
                if has_cycle(agent.name):
                    raise ValueError(
                        f"Circular dependency detected involving agent {agent.name}"
                    )

    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """Get agent by name."""
        return next((a for a in self.agents if a.name == name), None)

    def get_ready_agents(self) -> List[Agent]:
        """Get agents ready to execute (dependencies satisfied, not running/completed)."""
        completed = {
            a.name for a in self.agents
            if a.status == AgentStatus.COMPLETED
        }

        ready = []
        for agent in self.agents:
            if agent.status == AgentStatus.PENDING and agent.can_execute(completed):
                ready.append(agent)

        # Sort by priority (higher first)
        ready.sort(key=lambda a: a.priority, reverse=True)
        return ready

    def get_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics."""
        total = len(self.agents)
        completed = sum(1 for a in self.agents if a.status == AgentStatus.COMPLETED)
        failed = sum(1 for a in self.agents if a.status == AgentStatus.FAILED)
        running = sum(1 for a in self.agents if a.status == AgentStatus.RUNNING)
        pending = sum(1 for a in self.agents if a.status == AgentStatus.PENDING)

        return {
            "workflow_id": self.id,
            "workflow_name": self.name,
            "status": self.status.value,
            "total_agents": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "cost_budget": self.cost_budget,
            "budget_remaining": self.cost_budget - self.total_cost,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "module": self.module,
            "status": self.status.value,
            "agents": [a.to_dict() for a in self.agents],
            "statistics": self.get_statistics(),
        }

"""Base agent class and related types."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AgentStatus(Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class AgentResult:
    """Result of an agent execution."""
    agent_id: str
    status: AgentStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: int = 0
    cost: float = 0.0
    execution_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == AgentStatus.COMPLETED

    def is_retriable(self) -> bool:
        """Check if execution can be retried."""
        return self.status in [
            AgentStatus.FAILED,
            AgentStatus.RATE_LIMITED,
        ]


@dataclass
class Agent:
    """
    Base Agent class representing a single AI agent in the workflow.

    Attributes:
        name: Unique agent name
        module: ERP module (HR, Finance, Inventory, Compliance)
        description: Agent purpose
        llm_config: LLM configuration (model, temperature, etc.)
        tools: List of tools/functions available to the agent
        dependencies: List of agent names this agent depends on
        max_retries: Maximum retry attempts on failure
        timeout: Execution timeout in seconds
        cost_limit: Maximum cost allowed for this agent
        priority: Execution priority (higher = more important)
    """
    name: str
    module: str
    description: str
    llm_config: Dict[str, Any]
    prompt_template: str
    tools: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout: float = 300.0
    cost_limit: float = 1.0
    priority: int = 5

    # Runtime fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: AgentStatus = AgentStatus.PENDING
    result: Optional[AgentResult] = None

    # Callbacks
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate agent configuration."""
        if not self.name:
            raise ValueError("Agent name is required")
        if self.module not in ["HR", "Finance", "Inventory", "Compliance", "General"]:
            raise ValueError(f"Invalid module: {self.module}")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.cost_limit <= 0:
            raise ValueError("cost_limit must be > 0")

    def reset(self):
        """Reset agent state for re-execution."""
        self.status = AgentStatus.PENDING
        self.result = None

    def can_execute(self, completed_agents: set) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_agents for dep in self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "tags": self.tags,
        }

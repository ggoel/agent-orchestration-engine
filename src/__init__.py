"""
Agent Orchestration Engine
A production-ready system for managing 500+ AI agent workflows across ERP modules.
"""

__version__ = "1.0.0"
__author__ = "Agent Orchestration Team"

from .orchestrator.engine import OrchestrationEngine
from .agents.base import Agent, AgentResult, AgentStatus
from .agents.workflow import Workflow, WorkflowStatus

__all__ = [
    "OrchestrationEngine",
    "Agent",
    "AgentResult",
    "AgentStatus",
    "Workflow",
    "WorkflowStatus",
]

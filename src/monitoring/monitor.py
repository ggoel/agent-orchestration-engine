"""Monitoring and metrics collection."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MonitorConfig:
    """Monitoring configuration."""
    enable_metrics: bool = True
    enable_logging: bool = True
    metrics_interval: float = 60.0  # seconds


@dataclass
class MetricEvent:
    """Single metric event."""
    timestamp: datetime
    event_type: str
    agent_id: Optional[str]
    workflow_id: Optional[str]
    data: Dict[str, Any]


class Monitor:
    """
    System monitoring and observability.

    Collects metrics, logs, and provides real-time system state.
    """

    def __init__(self, config: MonitorConfig):
        """Initialize monitor."""
        self.config = config
        self.events: List[MetricEvent] = []
        self._lock = asyncio.Lock()

        # Metrics
        self.metrics = {
            "workflows_started": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "agents_executed": 0,
            "agents_failed": 0,
            "total_api_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_agent_execution_time": 0.0,
        }

        # Active workflows and agents
        self.active_workflows: Dict[str, Dict] = {}
        self.active_agents: Dict[str, Dict] = {}

        # Start time
        self.start_time = time.time()

    async def record_event(
        self,
        event_type: str,
        agent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        data: Optional[Dict] = None
    ):
        """Record a monitoring event."""
        if not self.config.enable_logging:
            return

        event = MetricEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            agent_id=agent_id,
            workflow_id=workflow_id,
            data=data or {}
        )

        async with self._lock:
            self.events.append(event)

            # Keep only last 10000 events
            if len(self.events) > 10000:
                self.events = self.events[-10000:]

    async def update_metric(self, metric_name: str, value: Any):
        """Update a metric value."""
        if not self.config.enable_metrics:
            return

        async with self._lock:
            if metric_name.startswith("total_") or metric_name.endswith("_count"):
                # Accumulate
                self.metrics[metric_name] = self.metrics.get(metric_name, 0) + value
            else:
                # Replace
                self.metrics[metric_name] = value

    async def increment_metric(self, metric_name: str, amount: int = 1):
        """Increment a counter metric."""
        await self.update_metric(metric_name, amount)

    async def record_workflow_start(self, workflow_id: str, workflow_data: Dict):
        """Record workflow start."""
        async with self._lock:
            self.active_workflows[workflow_id] = {
                "started_at": time.time(),
                "data": workflow_data
            }
            self.metrics["workflows_started"] += 1

        await self.record_event("workflow_start", workflow_id=workflow_id, data=workflow_data)

    async def record_workflow_complete(self, workflow_id: str, success: bool, stats: Dict):
        """Record workflow completion."""
        async with self._lock:
            if workflow_id in self.active_workflows:
                duration = time.time() - self.active_workflows[workflow_id]["started_at"]
                stats["duration"] = duration
                del self.active_workflows[workflow_id]

            if success:
                self.metrics["workflows_completed"] += 1
            else:
                self.metrics["workflows_failed"] += 1

        await self.record_event(
            "workflow_complete" if success else "workflow_failed",
            workflow_id=workflow_id,
            data=stats
        )

    async def record_agent_start(self, agent_id: str, workflow_id: str, agent_data: Dict):
        """Record agent execution start."""
        async with self._lock:
            self.active_agents[agent_id] = {
                "started_at": time.time(),
                "workflow_id": workflow_id,
                "data": agent_data
            }

        await self.record_event(
            "agent_start",
            agent_id=agent_id,
            workflow_id=workflow_id,
            data=agent_data
        )

    async def record_agent_complete(
        self,
        agent_id: str,
        workflow_id: str,
        success: bool,
        stats: Dict
    ):
        """Record agent execution completion."""
        async with self._lock:
            if agent_id in self.active_agents:
                duration = time.time() - self.active_agents[agent_id]["started_at"]
                stats["duration"] = duration
                del self.active_agents[agent_id]

            self.metrics["agents_executed"] += 1
            if not success:
                self.metrics["agents_failed"] += 1

            # Update average execution time
            current_avg = self.metrics.get("avg_agent_execution_time", 0.0)
            total_executed = self.metrics["agents_executed"]
            new_avg = (current_avg * (total_executed - 1) + stats.get("duration", 0)) / total_executed
            self.metrics["avg_agent_execution_time"] = new_avg

        await self.record_event(
            "agent_complete" if success else "agent_failed",
            agent_id=agent_id,
            workflow_id=workflow_id,
            data=stats
        )

    async def record_api_call(self, tokens: int, cost: float):
        """Record API call metrics."""
        async with self._lock:
            self.metrics["total_api_calls"] += 1
            self.metrics["total_tokens"] += tokens
            self.metrics["total_cost"] += cost

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        uptime = time.time() - self.start_time

        return {
            **self.metrics,
            "uptime_seconds": uptime,
            "active_workflows": len(self.active_workflows),
            "active_agents": len(self.active_agents),
            "events_recorded": len(self.events),
        }

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get status of active workflow."""
        if workflow_id not in self.active_workflows:
            return None

        workflow_data = self.active_workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "started_at": workflow_data["started_at"],
            "elapsed": time.time() - workflow_data["started_at"],
            "data": workflow_data["data"]
        }

    def get_recent_events(self, count: int = 100, event_type: Optional[str] = None) -> List[Dict]:
        """Get recent monitoring events."""
        events = self.events[-count:]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type,
                "agent_id": e.agent_id,
                "workflow_id": e.workflow_id,
                "data": e.data
            }
            for e in events
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        metrics = self.get_metrics()

        success_rate = 0.0
        if metrics["workflows_started"] > 0:
            success_rate = (
                metrics["workflows_completed"] /
                metrics["workflows_started"] * 100
            )

        agent_failure_rate = 0.0
        if metrics["agents_executed"] > 0:
            agent_failure_rate = (
                metrics["agents_failed"] /
                metrics["agents_executed"] * 100
            )

        return {
            "uptime_seconds": metrics["uptime_seconds"],
            "workflows": {
                "started": metrics["workflows_started"],
                "completed": metrics["workflows_completed"],
                "failed": metrics["workflows_failed"],
                "active": metrics["active_workflows"],
                "success_rate": success_rate,
            },
            "agents": {
                "executed": metrics["agents_executed"],
                "failed": metrics["agents_failed"],
                "active": metrics["active_agents"],
                "failure_rate": agent_failure_rate,
                "avg_execution_time": metrics["avg_agent_execution_time"],
            },
            "api": {
                "total_calls": metrics["total_api_calls"],
                "total_tokens": metrics["total_tokens"],
                "total_cost": metrics["total_cost"],
            }
        }

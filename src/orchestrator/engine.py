"""Main orchestration engine implementation."""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..agents.base import Agent, AgentResult, AgentStatus
from ..agents.workflow import Workflow, WorkflowStatus
from ..cost_tracking.tracker import CostTracker, CostConfig
from ..dependencies.resolver import DependencyResolver
from ..failure_handling.handler import FailureHandler, RetryPolicy
from ..llm_clients.base import LLMClient
from ..monitoring.monitor import Monitor, MonitorConfig
from ..rate_limiting.limiter import RateLimiter, RateLimitConfig


@dataclass
class EngineConfig:
    """Orchestration engine configuration."""
    rate_limit_config: RateLimitConfig
    cost_config: CostConfig
    monitor_config: MonitorConfig
    default_retry_policy: RetryPolicy
    max_concurrent_workflows: int = 50


class OrchestrationEngine:
    """
    Main orchestration engine for managing AI agent workflows.

    Coordinates execution of 500+ agents across multiple workflows with:
    - Rate limiting
    - Cost tracking and budget enforcement
    - Dependency resolution
    - Failure handling and retries
    - Real-time monitoring
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: EngineConfig
    ):
        """
        Initialize orchestration engine.

        Args:
            llm_client: LLM client for agent execution
            config: Engine configuration
        """
        self.llm_client = llm_client
        self.config = config

        # Core components
        self.rate_limiter = RateLimiter(config.rate_limit_config)
        self.cost_tracker = CostTracker(config.cost_config)
        self.failure_handler = FailureHandler(config.default_retry_policy)
        self.monitor = Monitor(config.monitor_config)

        # Workflow management
        self.workflows: Dict[str, Workflow] = {}
        self.workflow_semaphore = asyncio.Semaphore(config.max_concurrent_workflows)

        # Execution state
        self.running = False
        self._shutdown_event = asyncio.Event()

    async def register_workflow(self, workflow: Workflow) -> str:
        """
        Register a workflow for execution.

        Args:
            workflow: Workflow to register

        Returns:
            Workflow ID
        """
        self.workflows[workflow.id] = workflow

        # Set budget in cost tracker
        await self.cost_tracker.set_workflow_budget(workflow.id, workflow.cost_budget)

        # Set agent budgets
        for agent in workflow.agents:
            await self.cost_tracker.set_agent_budget(agent.id, agent.cost_limit)

        return workflow.id

    async def execute_workflow(self, workflow_id: str) -> Workflow:
        """
        Execute a workflow.

        Args:
            workflow_id: ID of workflow to execute

        Returns:
            Completed workflow with results
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]

        # Acquire workflow slot
        async with self.workflow_semaphore:
            return await self._execute_workflow_internal(workflow)

    async def _execute_workflow_internal(self, workflow: Workflow) -> Workflow:
        """Internal workflow execution logic."""
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()

        # Record workflow start
        await self.monitor.record_workflow_start(
            workflow.id,
            {"name": workflow.name, "agents": len(workflow.agents)}
        )

        try:
            # Resolve dependencies and get execution levels
            resolver = DependencyResolver(workflow.agents)
            execution_levels = resolver.get_execution_levels()

            # Execute agents level by level
            for level_idx, level_agents in enumerate(execution_levels):
                # Check budget before executing level
                if not await self.cost_tracker.check_workflow_budget(workflow.id):
                    workflow.status = WorkflowStatus.FAILED
                    raise Exception("Workflow budget exceeded")

                # Execute agents in level (in parallel)
                tasks = []
                for agent in level_agents[:workflow.max_parallel_agents]:
                    tasks.append(self._execute_agent(agent, workflow))

                # Wait for all agents in level to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle failures
                failed_agents = []
                for agent, result in zip(level_agents, results):
                    if isinstance(result, Exception):
                        agent.status = AgentStatus.FAILED
                        failed_agents.append(agent)
                    elif not result.is_success():
                        failed_agents.append(agent)

                # Check if workflow should fail
                if self.failure_handler.should_fail_workflow(
                    failed_agents,
                    len(workflow.agents),
                    workflow.allow_partial_completion
                ):
                    workflow.status = WorkflowStatus.FAILED
                    break

            # Determine final status
            if workflow.status != WorkflowStatus.FAILED:
                completed_count = sum(
                    1 for a in workflow.agents
                    if a.status == AgentStatus.COMPLETED
                )

                if completed_count == len(workflow.agents):
                    workflow.status = WorkflowStatus.COMPLETED
                else:
                    workflow.status = WorkflowStatus.PARTIALLY_COMPLETED

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            await self.monitor.record_event(
                "workflow_error",
                workflow_id=workflow.id,
                data={"error": str(e)}
            )

        finally:
            workflow.completed_at = datetime.utcnow()

            # Get final costs
            workflow.total_cost = await self.cost_tracker.get_workflow_cost(workflow.id)

            # Record workflow completion
            await self.monitor.record_workflow_complete(
                workflow.id,
                workflow.status == WorkflowStatus.COMPLETED,
                workflow.get_statistics()
            )

        return workflow

    async def _execute_agent(self, agent: Agent, workflow: Workflow) -> AgentResult:
        """
        Execute a single agent.

        Args:
            agent: Agent to execute
            workflow: Parent workflow

        Returns:
            Agent execution result
        """
        agent.status = AgentStatus.RUNNING
        start_time = time.time()

        # Record agent start
        await self.monitor.record_agent_start(
            agent.id,
            workflow.id,
            {"name": agent.name, "module": agent.module}
        )

        result = AgentResult(
            agent_id=agent.id,
            status=AgentStatus.PENDING,
            started_at=datetime.utcnow()
        )

        retry_count = 0
        while retry_count <= agent.max_retries:
            try:
                # Check agent budget
                if not await self.cost_tracker.check_agent_budget(agent.id):
                    result.status = AgentStatus.BUDGET_EXCEEDED
                    result.error = "Agent budget exceeded"
                    break

                # Acquire rate limit
                try:
                    await asyncio.wait_for(
                        self.rate_limiter.acquire(
                            estimated_tokens=agent.llm_config.get("max_tokens", 1000)
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    result.status = AgentStatus.RATE_LIMITED
                    result.error = "Rate limit timeout"
                    retry_count += 1
                    continue

                try:
                    # Execute agent
                    llm_result = await asyncio.wait_for(
                        self._call_llm(agent),
                        timeout=agent.timeout
                    )

                    # Record cost
                    cost = await self.cost_tracker.record_cost(
                        agent.id,
                        agent.name,
                        workflow.id,
                        agent.llm_config.get("model", "gpt-4"),
                        llm_result.input_tokens,
                        llm_result.output_tokens
                    )

                    # Update monitoring
                    await self.monitor.record_api_call(
                        llm_result.input_tokens + llm_result.output_tokens,
                        cost
                    )

                    # Build result
                    result.status = AgentStatus.COMPLETED
                    result.output = {
                        "content": llm_result.content,
                        "tool_calls": llm_result.tool_calls
                    }
                    result.tokens_used = llm_result.input_tokens + llm_result.output_tokens
                    result.cost = cost

                    # Call success callback
                    if agent.on_success:
                        try:
                            await agent.on_success(agent, result)
                        except Exception as e:
                            print(f"Error in success callback: {e}")

                    break

                finally:
                    self.rate_limiter.release()

            except asyncio.TimeoutError:
                result.status = AgentStatus.FAILED
                result.error = f"Agent execution timeout after {agent.timeout}s"
                retry_count += 1

            except Exception as e:
                result.status = AgentStatus.FAILED
                result.error = str(e)
                retry_count += 1

            # Handle failure
            if result.status != AgentStatus.COMPLETED:
                result.retry_count = retry_count

                # Check if should retry
                should_retry = await self.failure_handler.handle_failure(
                    agent,
                    result,
                    None  # Use default policy
                )

                if not should_retry:
                    break

        # Finalize result
        result.completed_at = datetime.utcnow()
        result.execution_time = time.time() - start_time
        agent.result = result
        agent.status = result.status

        # Record agent completion
        await self.monitor.record_agent_complete(
            agent.id,
            workflow.id,
            result.is_success(),
            {
                "status": result.status.value,
                "cost": result.cost,
                "tokens": result.tokens_used,
                "retries": result.retry_count
            }
        )

        return result

    async def _call_llm(self, agent: Agent) -> Any:
        """Call LLM for agent execution."""
        if agent.tools:
            # Use tools
            messages = [{"role": "user", "content": agent.prompt_template}]
            return await self.llm_client.complete_with_tools(
                messages=messages,
                model=agent.llm_config.get("model", "gpt-4"),
                tools=agent.tools,
                temperature=agent.llm_config.get("temperature", 0.7),
                max_tokens=agent.llm_config.get("max_tokens", 1000)
            )
        else:
            # Simple completion
            return await self.llm_client.complete(
                prompt=agent.prompt_template,
                model=agent.llm_config.get("model", "gpt-4"),
                temperature=agent.llm_config.get("temperature", 0.7),
                max_tokens=agent.llm_config.get("max_tokens", 1000)
            )

    async def execute_multiple_workflows(
        self,
        workflow_ids: List[str]
    ) -> List[Workflow]:
        """
        Execute multiple workflows in parallel.

        Args:
            workflow_ids: List of workflow IDs to execute

        Returns:
            List of completed workflows
        """
        tasks = [self.execute_workflow(wf_id) for wf_id in workflow_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        workflows = []
        for result in results:
            if isinstance(result, Workflow):
                workflows.append(result)
            else:
                print(f"Workflow execution error: {result}")

        return workflows

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get current status of a workflow."""
        if workflow_id not in self.workflows:
            return None

        workflow = self.workflows[workflow_id]
        return workflow.to_dict()

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "engine": {
                "running": self.running,
                "total_workflows": len(self.workflows),
            },
            "monitoring": self.monitor.get_summary(),
            "rate_limiting": self.rate_limiter.get_statistics(),
            "cost_tracking": self.cost_tracker.get_statistics(),
            "failure_handling": self.failure_handler.get_statistics(),
        }

    async def shutdown(self):
        """Gracefully shutdown the engine."""
        self.running = False
        self._shutdown_event.set()

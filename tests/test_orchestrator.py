"""
Unit tests for orchestration engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.agents.base import Agent, AgentStatus
from src.agents.workflow import Workflow, WorkflowStatus
from src.orchestrator.engine import OrchestrationEngine, EngineConfig
from src.llm_clients.base import LLMClient, LLMResponse
from src.rate_limiting.limiter import RateLimitConfig
from src.cost_tracking.tracker import CostConfig
from src.monitoring.monitor import MonitorConfig
from src.failure_handling.handler import RetryPolicy


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    async def complete(self, prompt, model, temperature=0.7, max_tokens=1000, tools=None, **kwargs):
        return LLMResponse(
            content="Test response",
            model=model,
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop",
            tool_calls=None,
            metadata={}
        )

    async def complete_with_tools(self, messages, model, tools, temperature=0.7, max_tokens=1000, **kwargs):
        return LLMResponse(
            content="Test response with tools",
            model=model,
            input_tokens=120,
            output_tokens=60,
            finish_reason="stop",
            tool_calls=[],
            metadata={}
        )

    def get_model_info(self, model):
        return {"max_tokens": 4096, "supports_tools": True}


@pytest.fixture
def engine_config():
    """Create test engine configuration."""
    return EngineConfig(
        rate_limit_config=RateLimitConfig(
            requests_per_minute=100,
            concurrent_requests=5
        ),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(enable_metrics=True),
        default_retry_policy=RetryPolicy(max_retries=2),
        max_concurrent_workflows=10
    )


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def engine(mock_llm_client, engine_config):
    """Create orchestration engine."""
    return OrchestrationEngine(llm_client=mock_llm_client, config=engine_config)


@pytest.fixture
def simple_agent():
    """Create a simple test agent."""
    return Agent(
        name="test_agent",
        module="Finance",
        description="Test agent",
        llm_config={"model": "gpt-3.5-turbo", "temperature": 0.5},
        prompt_template="Test prompt",
        dependencies=[]
    )


@pytest.fixture
def simple_workflow(simple_agent):
    """Create a simple test workflow."""
    return Workflow(
        name="test_workflow",
        description="Test workflow",
        agents=[simple_agent],
        cost_budget=1.0
    )


@pytest.mark.asyncio
async def test_register_workflow(engine, simple_workflow):
    """Test workflow registration."""
    workflow_id = await engine.register_workflow(simple_workflow)

    assert workflow_id == simple_workflow.id
    assert workflow_id in engine.workflows
    assert engine.workflows[workflow_id] == simple_workflow


@pytest.mark.asyncio
async def test_execute_simple_workflow(engine, simple_workflow):
    """Test executing a simple workflow."""
    workflow_id = await engine.register_workflow(simple_workflow)
    result = await engine.execute_workflow(workflow_id)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.total_cost > 0
    assert len(result.agents) == 1
    assert result.agents[0].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_with_dependencies():
    """Test workflow with agent dependencies."""
    agent1 = Agent(
        name="agent1",
        module="Finance",
        description="First agent",
        llm_config={"model": "gpt-3.5-turbo"},
        prompt_template="Agent 1",
        dependencies=[]
    )

    agent2 = Agent(
        name="agent2",
        module="Finance",
        description="Second agent",
        llm_config={"model": "gpt-3.5-turbo"},
        prompt_template="Agent 2",
        dependencies=["agent1"]
    )

    workflow = Workflow(
        name="dependency_test",
        description="Test dependencies",
        agents=[agent1, agent2],
        cost_budget=2.0
    )

    mock_client = MockLLMClient()
    config = EngineConfig(
        rate_limit_config=RateLimitConfig(concurrent_requests=5),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(),
        default_retry_policy=RetryPolicy()
    )
    engine = OrchestrationEngine(mock_client, config)

    workflow_id = await engine.register_workflow(workflow)
    result = await engine.execute_workflow(workflow_id)

    assert result.status == WorkflowStatus.COMPLETED
    assert agent1.status == AgentStatus.COMPLETED
    assert agent2.status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_budget_exceeded():
    """Test budget enforcement."""
    agent = Agent(
        name="expensive_agent",
        module="Finance",
        description="Expensive agent",
        llm_config={"model": "gpt-4"},
        prompt_template="Test",
        cost_limit=0.0001  # Very low limit
    )

    workflow = Workflow(
        name="budget_test",
        description="Test budget",
        agents=[agent],
        cost_budget=0.0001
    )

    mock_client = MockLLMClient()
    config = EngineConfig(
        rate_limit_config=RateLimitConfig(),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(),
        default_retry_policy=RetryPolicy()
    )
    engine = OrchestrationEngine(mock_client, config)

    workflow_id = await engine.register_workflow(workflow)
    result = await engine.execute_workflow(workflow_id)

    # Should fail due to budget
    assert result.status in [WorkflowStatus.FAILED, WorkflowStatus.PARTIALLY_COMPLETED]


@pytest.mark.asyncio
async def test_parallel_execution():
    """Test parallel agent execution."""
    agents = []
    for i in range(5):
        agent = Agent(
            name=f"parallel_agent_{i}",
            module="Finance",
            description=f"Agent {i}",
            llm_config={"model": "gpt-3.5-turbo"},
            prompt_template=f"Agent {i}",
            dependencies=[]
        )
        agents.append(agent)

    workflow = Workflow(
        name="parallel_test",
        description="Test parallel execution",
        agents=agents,
        max_parallel_agents=5,
        cost_budget=5.0
    )

    mock_client = MockLLMClient()
    config = EngineConfig(
        rate_limit_config=RateLimitConfig(concurrent_requests=10),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(),
        default_retry_policy=RetryPolicy()
    )
    engine = OrchestrationEngine(mock_client, config)

    workflow_id = await engine.register_workflow(workflow)
    result = await engine.execute_workflow(workflow_id)

    assert result.status == WorkflowStatus.COMPLETED
    assert all(a.status == AgentStatus.COMPLETED for a in result.agents)


@pytest.mark.asyncio
async def test_multiple_workflows(engine):
    """Test executing multiple workflows."""
    workflows = []
    for i in range(3):
        agent = Agent(
            name=f"agent_{i}",
            module="Finance",
            description=f"Agent {i}",
            llm_config={"model": "gpt-3.5-turbo"},
            prompt_template=f"Agent {i}"
        )
        workflow = Workflow(
            name=f"workflow_{i}",
            description=f"Workflow {i}",
            agents=[agent],
            cost_budget=1.0
        )
        workflows.append(workflow)

    workflow_ids = []
    for workflow in workflows:
        wf_id = await engine.register_workflow(workflow)
        workflow_ids.append(wf_id)

    results = await engine.execute_multiple_workflows(workflow_ids)

    assert len(results) == 3
    assert all(wf.status == WorkflowStatus.COMPLETED for wf in results)


@pytest.mark.asyncio
async def test_system_status(engine, simple_workflow):
    """Test getting system status."""
    workflow_id = await engine.register_workflow(simple_workflow)
    await engine.execute_workflow(workflow_id)

    status = engine.get_system_status()

    assert "engine" in status
    assert "monitoring" in status
    assert "rate_limiting" in status
    assert "cost_tracking" in status
    assert status["monitoring"]["workflows"]["completed"] >= 1


@pytest.mark.asyncio
async def test_workflow_status(engine, simple_workflow):
    """Test getting workflow status."""
    workflow_id = await engine.register_workflow(simple_workflow)

    # Before execution
    status = engine.get_workflow_status(workflow_id)
    assert status["status"] == WorkflowStatus.PENDING.value

    # After execution
    await engine.execute_workflow(workflow_id)
    status = engine.get_workflow_status(workflow_id)
    assert status["status"] == WorkflowStatus.COMPLETED.value


def test_circular_dependency_detection():
    """Test that circular dependencies are detected."""
    agent1 = Agent(
        name="agent1",
        module="Finance",
        description="Agent 1",
        llm_config={"model": "gpt-3.5-turbo"},
        prompt_template="Test",
        dependencies=["agent2"]
    )

    agent2 = Agent(
        name="agent2",
        module="Finance",
        description="Agent 2",
        llm_config={"model": "gpt-3.5-turbo"},
        prompt_template="Test",
        dependencies=["agent1"]
    )

    with pytest.raises(ValueError, match="Circular dependency"):
        workflow = Workflow(
            name="circular_test",
            description="Test circular dependency",
            agents=[agent1, agent2],
            cost_budget=1.0
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

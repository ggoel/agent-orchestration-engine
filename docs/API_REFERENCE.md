# API Reference

Complete API documentation for the Agent Orchestration Engine.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Orchestration Engine](#orchestration-engine)
3. [Agents](#agents)
4. [Workflows](#workflows)
5. [Rate Limiting](#rate-limiting)
6. [Cost Tracking](#cost-tracking)
7. [LLM Clients](#llm-clients)
8. [Monitoring](#monitoring)

---

## Core Classes

### Agent

Represents a single AI agent in the workflow.

```python
@dataclass
class Agent:
    """
    AI Agent configuration and state.

    Attributes:
        name (str): Unique agent identifier
        module (str): ERP module - "HR", "Finance", "Inventory", "Compliance", "General"
        description (str): Agent purpose
        llm_config (Dict[str, Any]): LLM configuration
        prompt_template (str): Prompt template for LLM
        tools (List[Dict]): Available tools/functions (optional)
        dependencies (List[str]): Names of agents this depends on (optional)
        max_retries (int): Maximum retry attempts (default: 3)
        timeout (float): Execution timeout in seconds (default: 300.0)
        cost_limit (float): Maximum cost in dollars (default: 1.0)
        priority (int): Execution priority 1-10 (default: 5)
    """
```

**Methods**:

- `reset() -> None`: Reset agent state for re-execution
- `can_execute(completed_agents: set) -> bool`: Check if dependencies satisfied
- `to_dict() -> Dict`: Convert to dictionary

**Example**:
```python
agent = Agent(
    name="data_processor",
    module="Finance",
    description="Process financial data",
    llm_config={
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 1000
    },
    prompt_template="Process the following data: {data}",
    dependencies=["data_validator"],
    priority=7
)
```

### AgentStatus

Enumeration of agent execution states.

```python
class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    BUDGET_EXCEEDED = "budget_exceeded"
```

### AgentResult

Result of an agent execution.

```python
@dataclass
class AgentResult:
    """
    Agent execution result.

    Attributes:
        agent_id (str): Agent identifier
        status (AgentStatus): Execution status
        output (Dict, optional): Agent output data
        error (str, optional): Error message if failed
        tokens_used (int): Total tokens consumed
        cost (float): Execution cost in dollars
        execution_time (float): Time taken in seconds
        started_at (datetime, optional): Start timestamp
        completed_at (datetime, optional): Completion timestamp
        retry_count (int): Number of retries attempted
        metadata (Dict): Additional metadata
    """
```

**Methods**:
- `is_success() -> bool`: Check if execution succeeded
- `is_retriable() -> bool`: Check if can be retried

### Workflow

Collection of agents with dependencies.

```python
@dataclass
class Workflow:
    """
    Workflow configuration.

    Attributes:
        name (str): Workflow name
        description (str): Workflow purpose
        agents (List[Agent]): Agents in workflow
        max_parallel_agents (int): Max parallel execution (default: 10)
        cost_budget (float): Maximum cost in dollars (default: 100.0)
        timeout (float): Workflow timeout in seconds (default: 3600.0)
        allow_partial_completion (bool): Continue on failures (default: True)
    """
```

**Methods**:
- `get_agent_by_name(name: str) -> Optional[Agent]`: Get agent by name
- `get_ready_agents() -> List[Agent]`: Get agents ready to execute
- `get_statistics() -> Dict`: Get execution statistics
- `to_dict() -> Dict`: Convert to dictionary

**Example**:
```python
workflow = Workflow(
    name="financial_analysis",
    description="Analyze financial data",
    agents=[agent1, agent2, agent3],
    max_parallel_agents=5,
    cost_budget=20.0,
    timeout=1800.0,
    allow_partial_completion=True
)
```

### WorkflowStatus

Enumeration of workflow states.

```python
class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

## Orchestration Engine

Main engine for workflow orchestration.

### OrchestrationEngine

```python
class OrchestrationEngine:
    """
    Main orchestration engine.

    Args:
        llm_client (LLMClient): LLM client for API calls
        config (EngineConfig): Engine configuration
    """
```

**Methods**:

#### `register_workflow(workflow: Workflow) -> str`

Register a workflow for execution.

**Parameters**:
- `workflow`: Workflow to register

**Returns**:
- `str`: Workflow ID

**Example**:
```python
workflow_id = await engine.register_workflow(workflow)
```

#### `execute_workflow(workflow_id: str) -> Workflow`

Execute a registered workflow.

**Parameters**:
- `workflow_id`: ID of workflow to execute

**Returns**:
- `Workflow`: Completed workflow with results

**Raises**:
- `ValueError`: If workflow not found

**Example**:
```python
result = await engine.execute_workflow(workflow_id)
print(f"Status: {result.status.value}")
print(f"Cost: ${result.total_cost:.4f}")
```

#### `execute_multiple_workflows(workflow_ids: List[str]) -> List[Workflow]`

Execute multiple workflows in parallel.

**Parameters**:
- `workflow_ids`: List of workflow IDs

**Returns**:
- `List[Workflow]`: List of completed workflows

**Example**:
```python
results = await engine.execute_multiple_workflows([id1, id2, id3])
```

#### `get_workflow_status(workflow_id: str) -> Optional[Dict]`

Get current workflow status.

**Parameters**:
- `workflow_id`: Workflow ID

**Returns**:
- `Dict` or `None`: Workflow status information

**Example**:
```python
status = engine.get_workflow_status(workflow_id)
print(status['status'])
```

#### `get_system_status() -> Dict[str, Any]`

Get overall system status and metrics.

**Returns**:
- `Dict`: System status including monitoring, rate limiting, costs

**Example**:
```python
status = engine.get_system_status()
print(f"Workflows: {status['monitoring']['workflows']['completed']}")
print(f"Total Cost: ${status['cost_tracking']['total_cost']:.4f}")
```

#### `shutdown() -> None`

Gracefully shutdown the engine.

**Example**:
```python
await engine.shutdown()
```

### EngineConfig

Engine configuration.

```python
@dataclass
class EngineConfig:
    """
    Engine configuration.

    Attributes:
        rate_limit_config (RateLimitConfig): Rate limiting configuration
        cost_config (CostConfig): Cost tracking configuration
        monitor_config (MonitorConfig): Monitoring configuration
        default_retry_policy (RetryPolicy): Default retry policy
        max_concurrent_workflows (int): Max concurrent workflows (default: 50)
    """
```

**Example**:
```python
config = EngineConfig(
    rate_limit_config=RateLimitConfig(
        requests_per_minute=60,
        concurrent_requests=10
    ),
    cost_config=CostConfig(),
    monitor_config=MonitorConfig(),
    default_retry_policy=RetryPolicy(max_retries=3),
    max_concurrent_workflows=50
)
```

---

## Rate Limiting

### RateLimiter

Multi-tier rate limiter for API calls.

```python
class RateLimiter:
    """
    Rate limiter with token bucket algorithm.

    Args:
        config (RateLimitConfig): Rate limit configuration
    """
```

**Methods**:

#### `acquire(estimated_tokens: int = 1000, timeout: Optional[float] = None) -> None`

Acquire permission to make API call.

**Parameters**:
- `estimated_tokens`: Estimated token usage
- `timeout`: Maximum wait time in seconds

**Raises**:
- `asyncio.TimeoutError`: If timeout exceeded

**Example**:
```python
await rate_limiter.acquire(estimated_tokens=1500, timeout=30.0)
try:
    # Make API call
    pass
finally:
    rate_limiter.release()
```

#### `release() -> None`

Release concurrent request slot.

#### `get_statistics() -> Dict`

Get rate limiter statistics.

**Returns**:
- `Dict`: Statistics including total requests, tokens, wait times

### RateLimitConfig

Rate limit configuration.

```python
@dataclass
class RateLimitConfig:
    """
    Rate limit configuration.

    Attributes:
        requests_per_minute (int): Max requests per minute (default: 60)
        requests_per_hour (int): Max requests per hour (default: 3000)
        requests_per_day (int): Max requests per day (default: 50000)
        tokens_per_minute (int): Max tokens per minute (default: 100000)
        concurrent_requests (int): Max concurrent requests (default: 10)
    """
```

---

## Cost Tracking

### CostTracker

Tracks costs and enforces budgets.

```python
class CostTracker:
    """
    Cost tracking and budget management.

    Args:
        config (CostConfig): Cost configuration
    """
```

**Methods**:

#### `calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float`

Calculate cost for a model invocation.

**Parameters**:
- `model`: Model name
- `input_tokens`: Number of input tokens
- `output_tokens`: Number of output tokens

**Returns**:
- `float`: Cost in dollars

**Example**:
```python
cost = cost_tracker.calculate_cost("gpt-4", 1000, 500)
print(f"Estimated cost: ${cost:.4f}")
```

#### `record_cost(...) -> float`

Record cost for agent execution.

**Parameters**:
- `agent_id`: Agent ID
- `agent_name`: Agent name
- `workflow_id`: Workflow ID
- `model`: Model used
- `input_tokens`: Input token count
- `output_tokens`: Output token count
- `metadata`: Optional metadata

**Returns**:
- `float`: Cost for this execution

#### `check_workflow_budget(workflow_id: str) -> bool`

Check if workflow is within budget.

**Returns**:
- `bool`: True if within budget

#### `check_agent_budget(agent_id: str) -> bool`

Check if agent is within budget.

**Returns**:
- `bool`: True if within budget

#### `get_statistics() -> Dict`

Get cost statistics.

### CostConfig

Cost configuration.

```python
@dataclass
class CostConfig:
    """
    Cost configuration.

    Attributes:
        model_costs (Dict): Cost per 1K tokens for each model
        default_input_cost (float): Default input cost per 1K tokens
        default_output_cost (float): Default output cost per 1K tokens
    """
```

**Example**:
```python
config = CostConfig(
    model_costs={
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
    }
)
```

---

## LLM Clients

### LLMClient

Abstract base class for LLM providers.

```python
class LLMClient(ABC):
    """Base class for LLM clients."""
```

**Methods**:

#### `complete(prompt, model, temperature, max_tokens, tools, **kwargs) -> LLMResponse`

Generate completion from LLM.

**Parameters**:
- `prompt`: Input prompt
- `model`: Model name
- `temperature`: Sampling temperature (0-1)
- `max_tokens`: Maximum tokens to generate
- `tools`: Available tools (optional)
- `**kwargs`: Additional parameters

**Returns**:
- `LLMResponse`: Completion and metadata

#### `complete_with_tools(messages, model, tools, ...) -> LLMResponse`

Generate completion with tool use.

**Parameters**:
- `messages`: Conversation messages
- `model`: Model name
- `tools`: Available tools
- Additional parameters

**Returns**:
- `LLMResponse`: Completion with tool calls

### OpenAIClient

OpenAI implementation.

```python
class OpenAIClient(LLMClient):
    """
    OpenAI API client.

    Args:
        api_key (str): OpenAI API key
    """
```

**Example**:
```python
client = OpenAIClient(api_key="sk-...")

response = await client.complete(
    prompt="Analyze this data...",
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

print(response.content)
print(f"Tokens: {response.input_tokens + response.output_tokens}")
```

### AnthropicClient

Anthropic Claude implementation.

```python
class AnthropicClient(LLMClient):
    """
    Anthropic Claude API client.

    Args:
        api_key (str): Anthropic API key
    """
```

### LLMResponse

Response from LLM API.

```python
@dataclass
class LLMResponse:
    """
    LLM API response.

    Attributes:
        content (str): Generated text
        model (str): Model used
        input_tokens (int): Input token count
        output_tokens (int): Output token count
        finish_reason (str): Completion reason
        tool_calls (List, optional): Tool calls if any
        metadata (Dict): Additional metadata
    """
```

---

## Monitoring

### Monitor

System monitoring and metrics.

```python
class Monitor:
    """
    Monitoring and observability.

    Args:
        config (MonitorConfig): Monitoring configuration
    """
```

**Methods**:

#### `record_event(event_type, agent_id, workflow_id, data) -> None`

Record a monitoring event.

**Parameters**:
- `event_type`: Type of event
- `agent_id`: Agent ID (optional)
- `workflow_id`: Workflow ID (optional)
- `data`: Event data (optional)

#### `get_metrics() -> Dict`

Get current metrics snapshot.

**Returns**:
- `Dict`: All collected metrics

**Example**:
```python
metrics = monitor.get_metrics()
print(f"Uptime: {metrics['uptime_seconds']:.0f}s")
print(f"Total workflows: {metrics['workflows_started']}")
print(f"Total cost: ${metrics['total_cost']:.4f}")
```

#### `get_summary() -> Dict`

Get comprehensive system summary.

**Returns**:
- `Dict`: Detailed system statistics

**Example**:
```python
summary = monitor.get_summary()
print(f"Success rate: {summary['workflows']['success_rate']:.1f}%")
print(f"Avg execution time: {summary['agents']['avg_execution_time']:.2f}s")
```

#### `get_recent_events(count: int = 100, event_type: Optional[str] = None) -> List[Dict]`

Get recent monitoring events.

**Parameters**:
- `count`: Number of events to return
- `event_type`: Filter by event type (optional)

**Returns**:
- `List[Dict]`: Recent events

### MonitorConfig

Monitoring configuration.

```python
@dataclass
class MonitorConfig:
    """
    Monitoring configuration.

    Attributes:
        enable_metrics (bool): Enable metrics collection (default: True)
        enable_logging (bool): Enable event logging (default: True)
        metrics_interval (float): Metrics collection interval (default: 60.0)
    """
```

---

## Failure Handling

### FailureHandler

Handles failures and retries.

```python
class FailureHandler:
    """
    Failure handling and retry logic.

    Args:
        default_policy (RetryPolicy, optional): Default retry policy
        on_failure (Callable, optional): Failure callback
    """
```

**Methods**:

#### `handle_failure(agent, result, policy) -> bool`

Handle agent failure.

**Parameters**:
- `agent`: Failed agent
- `result`: Agent result with error
- `policy`: Retry policy (uses default if None)

**Returns**:
- `bool`: True if should retry, False otherwise

#### `should_fail_workflow(failed_agents, total_agents, allow_partial) -> bool`

Determine if workflow should fail.

**Parameters**:
- `failed_agents`: List of failed agents
- `total_agents`: Total agent count
- `allow_partial_completion`: Allow partial completion

**Returns**:
- `bool`: True if workflow should fail

### RetryPolicy

Retry policy configuration.

```python
@dataclass
class RetryPolicy:
    """
    Retry policy configuration.

    Attributes:
        max_retries (int): Maximum retry attempts (default: 3)
        strategy (RetryStrategy): Retry strategy (default: EXPONENTIAL_BACKOFF)
        base_delay (float): Base delay in seconds (default: 1.0)
        max_delay (float): Maximum delay in seconds (default: 60.0)
        exponential_base (float): Exponential base (default: 2.0)
        jitter (bool): Add jitter to delays (default: True)
    """
```

**Example**:
```python
policy = RetryPolicy(
    max_retries=5,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay=2.0,
    max_delay=120.0,
    jitter=True
)
```

### RetryStrategy

Retry strategy enumeration.

```python
class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CONSTANT_DELAY = "constant_delay"
    IMMEDIATE = "immediate"
```

---

## Complete Example

```python
import asyncio
from src.orchestrator.engine import OrchestrationEngine, EngineConfig
from src.agents.base import Agent
from src.agents.workflow import Workflow
from src.llm_clients.openai_client import OpenAIClient
from src.rate_limiting.limiter import RateLimitConfig
from src.cost_tracking.tracker import CostConfig
from src.monitoring.monitor import MonitorConfig
from src.failure_handling.handler import RetryPolicy

async def main():
    # Initialize client
    llm_client = OpenAIClient(api_key="your-key")

    # Configure engine
    config = EngineConfig(
        rate_limit_config=RateLimitConfig(requests_per_minute=60),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(),
        default_retry_policy=RetryPolicy(max_retries=3)
    )

    # Create engine
    engine = OrchestrationEngine(llm_client, config)

    # Define workflow
    agent = Agent(
        name="analyzer",
        module="Finance",
        description="Analyze data",
        llm_config={"model": "gpt-4"},
        prompt_template="Analyze..."
    )

    workflow = Workflow(
        name="analysis",
        description="Data analysis",
        agents=[agent],
        cost_budget=5.0
    )

    # Execute
    wf_id = await engine.register_workflow(workflow)
    result = await engine.execute_workflow(wf_id)

    # Get results
    print(f"Status: {result.status.value}")
    print(f"Cost: ${result.total_cost:.4f}")

    # Get system status
    status = engine.get_system_status()
    print(f"Total calls: {status['rate_limiting']['total_requests']}")

asyncio.run(main())
```

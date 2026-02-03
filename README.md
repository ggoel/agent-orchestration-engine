# Agent Orchestration Engine

A production-ready system for orchestrating 500+ AI agent workflows across multiple ERP modules (HR, Finance, Inventory, Compliance) with advanced rate limiting, cost tracking, and failure handling.

## Features

### Core Capabilities
- **Massive Scale**: Handle 500+ concurrent AI agents across multiple workflows
- **Multi-Module Support**: HR, Finance, Inventory, Compliance, and custom modules
- **Dependency Management**: Complex agent dependencies with automatic resolution
- **Rate Limiting**: Multi-tier rate limiting (per minute, hour, day) with token bucket algorithm
- **Cost Tracking**: Real-time cost tracking with budget enforcement at workflow and agent levels
- **Failure Handling**: Intelligent retry policies with exponential backoff
- **Monitoring**: Comprehensive metrics, logging, and observability
- **Tool Use**: Full support for LLM function calling and tool use
- **Async/Await**: Built on asyncio for high-performance concurrent execution

### Advanced Features
- **Partial Completion**: Workflows can complete partially with non-critical failures
- **Priority Scheduling**: Priority-based agent execution
- **Dependency Resolution**: Automatic topological sorting and parallel execution optimization
- **Cross-Module Dependencies**: Agents can depend on agents from other modules
- **Budget Enforcement**: Prevent cost overruns with configurable budgets
- **Multiple LLM Providers**: Support for OpenAI, Anthropic Claude, and extensible to others

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Or install from source:

```bash
pip install -e .
```

### Basic Usage

```python
import asyncio
from src.agents.base import Agent
from src.agents.workflow import Workflow
from src.orchestrator.engine import OrchestrationEngine, EngineConfig
from src.llm_clients.openai_client import OpenAIClient
from src.rate_limiting.limiter import RateLimitConfig
from src.cost_tracking.tracker import CostConfig
from src.monitoring.monitor import MonitorConfig
from src.failure_handling.handler import RetryPolicy

async def main():
    # Initialize LLM client
    llm_client = OpenAIClient(api_key="your-api-key")

    # Configure engine
    config = EngineConfig(
        rate_limit_config=RateLimitConfig(
            requests_per_minute=60,
            concurrent_requests=10
        ),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(),
        default_retry_policy=RetryPolicy(max_retries=3)
    )

    # Create engine
    engine = OrchestrationEngine(llm_client=llm_client, config=config)

    # Define agents
    agent1 = Agent(
        name="data_processor",
        module="Finance",
        description="Process financial data",
        llm_config={"model": "gpt-4", "temperature": 0.3},
        prompt_template="Process the financial data...",
        dependencies=[]
    )

    agent2 = Agent(
        name="report_generator",
        module="Finance",
        description="Generate report",
        llm_config={"model": "gpt-4", "temperature": 0.5},
        prompt_template="Generate a report...",
        dependencies=["data_processor"]
    )

    # Create workflow
    workflow = Workflow(
        name="financial_analysis",
        description="Financial data analysis workflow",
        agents=[agent1, agent2],
        cost_budget=10.0
    )

    # Execute
    workflow_id = await engine.register_workflow(workflow)
    result = await engine.execute_workflow(workflow_id)

    print(f"Status: {result.status.value}")
    print(f"Cost: ${result.total_cost:.4f}")

asyncio.run(main())
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│              Orchestration Engine                       │
│  - Workflow Management                                  │
│  - Execution Coordination                               │
│  - Component Integration                                │
└──────────────┬──────────────────────────────────────────┘
               │
    ┌──────────┼──────────┬──────────┬────────────┐
    │          │          │          │            │
┌───▼───┐  ┌──▼───┐  ┌──▼───┐  ┌───▼───┐  ┌────▼─────┐
│ Rate  │  │ Cost │  │Depend│  │Failure│  │Monitoring│
│Limiter│  │Track │  │Resolv│  │Handler│  │          │
└───────┘  └──────┘  └──────┘  └───────┘  └──────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼───┐  ┌──▼───┐
│OpenAI │  │Claude│  │Custom│
│Client │  │Client│  │Client│
└───────┘  └──────┘  └──────┘
```

### Key Components

1. **Orchestration Engine** (`src/orchestrator/engine.py`)
   - Central coordinator for all agent executions
   - Manages workflow lifecycle
   - Integrates all subsystems

2. **Agent System** (`src/agents/`)
   - Agent definition and configuration
   - Workflow creation and validation
   - Dependency management

3. **Rate Limiting** (`src/rate_limiting/`)
   - Token bucket algorithm
   - Multi-tier limiting (minute, hour, day)
   - Concurrent request limiting

4. **Cost Tracking** (`src/cost_tracking/`)
   - Real-time cost calculation
   - Budget enforcement
   - Per-workflow and per-agent tracking

5. **Dependency Resolution** (`src/dependencies/`)
   - Topological sorting
   - Parallel execution optimization
   - Cycle detection

6. **Failure Handling** (`src/failure_handling/`)
   - Retry policies
   - Exponential backoff
   - Partial failure support

7. **Monitoring** (`src/monitoring/`)
   - Metrics collection
   - Event logging
   - System observability

8. **LLM Clients** (`src/llm_clients/`)
   - Abstraction layer for LLM providers
   - OpenAI and Anthropic support
   - Extensible for custom providers

## Examples

### 1. Basic Workflow
See `examples/basic_workflow.py` for a simple three-agent workflow.

### 2. Multi-Module ERP Workflow
See `examples/erp_multi_module.py` for a complex workflow with 11+ agents across 4 modules.

### 3. Parallel Workflows
See `examples/parallel_workflows.py` for executing multiple workflows concurrently.

## Configuration

### Rate Limiting

```python
RateLimitConfig(
    requests_per_minute=60,      # Max requests per minute
    requests_per_hour=3000,      # Max requests per hour
    requests_per_day=50000,      # Max requests per day
    tokens_per_minute=100000,    # Max tokens per minute
    concurrent_requests=10       # Max concurrent requests
)
```

### Cost Configuration

```python
CostConfig(
    model_costs={
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        # Add custom models
    }
)
```

### Retry Policy

```python
RetryPolicy(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True
)
```

## Agent Configuration

### Basic Agent

```python
Agent(
    name="unique_agent_name",
    module="Finance",
    description="Agent purpose",
    llm_config={
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    prompt_template="Your prompt here...",
    dependencies=["other_agent"],
    max_retries=3,
    timeout=300.0,
    cost_limit=1.0,
    priority=5
)
```

### Agent with Tools

```python
Agent(
    name="tool_user",
    module="Finance",
    description="Agent that uses tools",
    llm_config={"model": "gpt-4"},
    prompt_template="Process data using available tools",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "calculate_sum",
                "description": "Calculate sum of numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    }
                }
            }
        }
    ]
)
```

## Workflow Configuration

```python
Workflow(
    name="workflow_name",
    description="Workflow purpose",
    agents=[agent1, agent2, agent3],
    max_parallel_agents=10,      # Max agents to run in parallel
    cost_budget=100.0,           # Maximum cost in dollars
    timeout=3600.0,              # Timeout in seconds
    allow_partial_completion=True, # Continue on non-critical failures
    module="Finance"
)
```

## Monitoring and Metrics

### Get System Status

```python
status = engine.get_system_status()
print(status)
```

Output:
```python
{
    "engine": {
        "running": True,
        "total_workflows": 10
    },
    "monitoring": {
        "workflows": {
            "started": 10,
            "completed": 8,
            "failed": 2,
            "success_rate": 80.0
        },
        "agents": {
            "executed": 50,
            "failed": 3,
            "avg_execution_time": 2.5
        }
    },
    "rate_limiting": {
        "total_requests": 50,
        "total_tokens": 45000,
        "avg_wait_time": 0.1
    },
    "cost_tracking": {
        "total_cost": 5.75,
        "total_tokens": 45000
    }
}
```

### Get Workflow Status

```python
workflow_status = engine.get_workflow_status(workflow_id)
```

## Best Practices

### 1. Dependency Design
- Keep dependency chains shallow for better parallelization
- Use priorities to indicate critical agents
- Avoid circular dependencies

### 2. Cost Management
- Set realistic budgets at workflow and agent levels
- Use cheaper models (gpt-3.5-turbo) for non-critical tasks
- Monitor costs in real-time

### 3. Rate Limiting
- Configure limits based on your API tier
- Leave headroom for rate limit spikes
- Use concurrent_requests to control parallelism

### 4. Failure Handling
- Set max_retries based on task criticality
- Use allow_partial_completion for non-critical workflows
- Implement custom callbacks for failure handling

### 5. Monitoring
- Regularly check system metrics
- Monitor average execution times
- Track cost trends

## Performance

### Throughput
- **500+ agents**: Can orchestrate 500+ concurrent agents
- **Parallel execution**: Automatic parallelization based on dependencies
- **Rate limit aware**: Efficiently manages API rate limits

### Scalability
- **Horizontal scaling**: Can run multiple engine instances
- **Async I/O**: Non-blocking I/O for maximum throughput
- **Resource efficient**: Minimal memory footprint

### Cost Optimization
- **Budget enforcement**: Prevent runaway costs
- **Model selection**: Support for cost-efficient models
- **Token tracking**: Accurate token usage tracking

## Testing

Run tests:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=src tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or contributions:
- GitHub Issues: [github.com/yourorg/agent-orchestration-engine/issues](https://github.com/yourorg/agent-orchestration-engine/issues)
- Documentation: [docs/](docs/)
- Examples: [examples/](examples/)

## Roadmap

- [ ] Distributed execution support
- [ ] Redis-based rate limiting for multi-instance deployments
- [ ] Prometheus metrics export
- [ ] Web dashboard for monitoring
- [ ] Workflow templates library
- [ ] Agent marketplace
- [ ] Enhanced tool use patterns
- [ ] Streaming responses support

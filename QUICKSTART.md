# Quick Start Guide

Get started with the Agent Orchestration Engine in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/agent-orchestration-engine.git
cd agent-orchestration-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Set Up API Key

```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Or export directly
export OPENAI_API_KEY="your-api-key-here"
```

## Run Your First Workflow

### 1. Create a Simple Script

Create `my_first_workflow.py`:

```python
import asyncio
import os
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
    api_key = os.getenv("OPENAI_API_KEY")
    llm_client = OpenAIClient(api_key=api_key)

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

    # Define agent
    agent = Agent(
        name="summarizer",
        module="General",
        description="Summarize text",
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_tokens": 500
        },
        prompt_template="Summarize this text: The Agent Orchestration Engine is a powerful tool for managing AI agents."
    )

    # Create workflow
    workflow = Workflow(
        name="simple_summary",
        description="Simple summarization workflow",
        agents=[agent],
        cost_budget=1.0
    )

    # Execute
    print("Registering workflow...")
    workflow_id = await engine.register_workflow(workflow)

    print("Executing workflow...")
    result = await engine.execute_workflow(workflow_id)

    # Print results
    print(f"\n{'='*50}")
    print(f"Workflow Status: {result.status.value}")
    print(f"Total Cost: ${result.total_cost:.4f}")
    print(f"{'='*50}\n")

    for agent in result.agents:
        print(f"Agent: {agent.name}")
        print(f"  Status: {agent.status.value}")
        if agent.result and agent.result.output:
            print(f"  Output: {agent.result.output['content']}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Run It

```bash
python my_first_workflow.py
```

Output:
```
Registering workflow...
Executing workflow...

==================================================
Workflow Status: completed
Total Cost: $0.0023
==================================================

Agent: summarizer
  Status: completed
  Output: The Agent Orchestration Engine is a powerful tool...
```

## Run Example Workflows

### Basic Workflow
```bash
python examples/basic_workflow.py
```

### Multi-Module ERP Workflow
```bash
python examples/erp_multi_module.py
```

### Parallel Workflows
```bash
python examples/parallel_workflows.py
```

## Next Steps

1. **Read the Architecture**: See `docs/ARCHITECTURE.md` for detailed system design
2. **API Reference**: Check `docs/API_REFERENCE.md` for complete API documentation
3. **Deployment Guide**: See `docs/DEPLOYMENT.md` for production deployment
4. **Customize**: Modify `config/example_config.yaml` for your needs

## Common Use Cases

### 1. Create an Agent with Dependencies

```python
agent1 = Agent(
    name="data_extractor",
    module="Finance",
    description="Extract data",
    llm_config={"model": "gpt-4"},
    prompt_template="Extract financial data..."
)

agent2 = Agent(
    name="data_analyzer",
    module="Finance",
    description="Analyze data",
    llm_config={"model": "gpt-4"},
    prompt_template="Analyze the extracted data...",
    dependencies=["data_extractor"]  # Depends on agent1
)
```

### 2. Use Tools (Function Calling)

```python
agent = Agent(
    name="calculator",
    module="Finance",
    description="Calculate values",
    llm_config={"model": "gpt-4"},
    prompt_template="Calculate the sum of 10 and 20",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    }
                }
            }
        }
    ]
)
```

### 3. Set Budgets

```python
# Workflow budget
workflow = Workflow(
    name="budgeted_workflow",
    agents=[agent1, agent2],
    cost_budget=5.0  # Max $5 for entire workflow
)

# Agent budget
agent = Agent(
    name="budgeted_agent",
    module="Finance",
    llm_config={"model": "gpt-4"},
    prompt_template="Process data",
    cost_limit=1.0  # Max $1 for this agent
)
```

### 4. Configure Retries

```python
config = EngineConfig(
    default_retry_policy=RetryPolicy(
        max_retries=5,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=2.0,
        max_delay=120.0
    ),
    # ... other config
)
```

### 5. Monitor Execution

```python
# Execute workflow
result = await engine.execute_workflow(workflow_id)

# Get statistics
stats = result.get_statistics()
print(f"Completed: {stats['completed']}/{stats['total_agents']}")
print(f"Cost: ${stats['total_cost']:.4f}")

# Get system status
status = engine.get_system_status()
print(f"Total API Calls: {status['rate_limiting']['total_requests']}")
```

## Troubleshooting

### Issue: API Key Error

```
Error: Invalid API key
```

**Solution**: Make sure your API key is set correctly:
```bash
export OPENAI_API_KEY="sk-your-actual-key"
```

### Issue: Rate Limit Exceeded

```
Status: rate_limited
```

**Solution**: Reduce rate limits in config:
```python
rate_limit_config=RateLimitConfig(
    requests_per_minute=30,  # Lower limit
    concurrent_requests=5     # Fewer concurrent requests
)
```

### Issue: Budget Exceeded

```
Status: budget_exceeded
```

**Solution**: Increase budget or use cheaper models:
```python
workflow.cost_budget = 10.0  # Increase budget

# Or use cheaper model
agent.llm_config = {"model": "gpt-3.5-turbo"}
```

## Tips for Success

1. **Start Small**: Begin with a single agent workflow
2. **Monitor Costs**: Always set budgets to prevent surprises
3. **Test First**: Use development rate limits before production
4. **Read Logs**: Enable detailed logging for debugging
5. **Check Examples**: Refer to `examples/` for patterns

## Support

- **Documentation**: Full docs in `docs/` folder
- **Examples**: Working examples in `examples/` folder
- **Issues**: GitHub issues for bug reports
- **API Reference**: See `docs/API_REFERENCE.md`

## What's Next?

- Build complex multi-agent workflows
- Integrate with your ERP system
- Deploy to production
- Scale to hundreds of agents
- Monitor and optimize performance

Happy orchestrating! 🚀

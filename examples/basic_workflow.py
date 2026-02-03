"""
Basic workflow example demonstrating agent orchestration.
"""

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
    llm_client = OpenAIClient(api_key="your-api-key-here")

    # Configure engine
    config = EngineConfig(
        rate_limit_config=RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=3000,
            tokens_per_minute=100000,
            concurrent_requests=10
        ),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(enable_metrics=True),
        default_retry_policy=RetryPolicy(max_retries=3),
        max_concurrent_workflows=50
    )

    # Create orchestration engine
    engine = OrchestrationEngine(llm_client=llm_client, config=config)

    # Define agents
    agent1 = Agent(
        name="data_extractor",
        module="Finance",
        description="Extract financial data from documents",
        llm_config={
            "model": "gpt-4",
            "temperature": 0.3,
            "max_tokens": 1000
        },
        prompt_template="Extract key financial metrics from the provided document.",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_numbers",
                    "description": "Extract numerical values from text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        }
                    }
                }
            }
        ],
        dependencies=[],
        priority=8
    )

    agent2 = Agent(
        name="data_validator",
        module="Finance",
        description="Validate extracted financial data",
        llm_config={
            "model": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 800
        },
        prompt_template="Validate the financial data for accuracy and completeness.",
        dependencies=["data_extractor"],
        priority=7
    )

    agent3 = Agent(
        name="report_generator",
        module="Finance",
        description="Generate financial report",
        llm_config={
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 2000
        },
        prompt_template="Generate a comprehensive financial report based on validated data.",
        dependencies=["data_validator"],
        priority=6
    )

    # Create workflow
    workflow = Workflow(
        name="financial_analysis",
        description="Extract, validate, and report financial data",
        agents=[agent1, agent2, agent3],
        max_parallel_agents=5,
        cost_budget=10.0,
        timeout=1800.0,
        allow_partial_completion=False,
        module="Finance"
    )

    # Register and execute workflow
    workflow_id = await engine.register_workflow(workflow)
    print(f"Registered workflow: {workflow_id}")

    print("Executing workflow...")
    result = await engine.execute_workflow(workflow_id)

    # Print results
    print(f"\nWorkflow Status: {result.status.value}")
    print(f"Total Cost: ${result.total_cost:.4f}")
    print(f"Total Tokens: {result.total_tokens}")

    print("\nAgent Results:")
    for agent in result.agents:
        print(f"\n  {agent.name}:")
        print(f"    Status: {agent.status.value}")
        if agent.result:
            print(f"    Cost: ${agent.result.cost:.4f}")
            print(f"    Tokens: {agent.result.tokens_used}")
            print(f"    Execution Time: {agent.result.execution_time:.2f}s")

    # Get system status
    print("\n" + "="*50)
    print("System Status:")
    status = engine.get_system_status()
    print(f"  Workflows Completed: {status['monitoring']['workflows']['completed']}")
    print(f"  Total Cost: ${status['cost_tracking']['total_cost']:.4f}")
    print(f"  Total API Calls: {status['rate_limiting']['total_requests']}")


if __name__ == "__main__":
    asyncio.run(main())

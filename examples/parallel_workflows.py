"""
Example demonstrating parallel execution of multiple workflows.
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


def create_workflow(name: str, module: str, agent_count: int) -> Workflow:
    """Create a workflow with specified number of agents."""
    agents = []
    for i in range(agent_count):
        dependencies = [f"{name}_agent_{i-1}"] if i > 0 else []

        agent = Agent(
            name=f"{name}_agent_{i}",
            module=module,
            description=f"Agent {i} for {name} workflow",
            llm_config={"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 500},
            prompt_template=f"Process step {i} of {name}",
            dependencies=dependencies,
            priority=5
        )
        agents.append(agent)

    return Workflow(
        name=name,
        description=f"{module} workflow: {name}",
        agents=agents,
        max_parallel_agents=5,
        cost_budget=5.0,
        timeout=1800.0,
        allow_partial_completion=True,
        module=module
    )


async def main():
    # Initialize
    llm_client = OpenAIClient(api_key="your-api-key-here")

    config = EngineConfig(
        rate_limit_config=RateLimitConfig(
            requests_per_minute=200,
            concurrent_requests=50
        ),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(enable_metrics=True),
        default_retry_policy=RetryPolicy(max_retries=2),
        max_concurrent_workflows=100
    )

    engine = OrchestrationEngine(llm_client=llm_client, config=config)

    # Create multiple workflows for different modules
    workflows = [
        create_workflow("hr_onboarding", "HR", 5),
        create_workflow("payroll_processing", "HR", 4),
        create_workflow("invoice_processing", "Finance", 6),
        create_workflow("expense_approval", "Finance", 4),
        create_workflow("stock_replenishment", "Inventory", 5),
        create_workflow("warehouse_optimization", "Inventory", 3),
        create_workflow("compliance_check", "Compliance", 4),
        create_workflow("audit_preparation", "Compliance", 5),
    ]

    print(f"Created {len(workflows)} workflows")
    print(f"Total agents: {sum(len(wf.agents) for wf in workflows)}")

    # Register all workflows
    workflow_ids = []
    for workflow in workflows:
        wf_id = await engine.register_workflow(workflow)
        workflow_ids.append(wf_id)

    print("\nExecuting all workflows in parallel...")
    start_time = asyncio.get_event_loop().time()

    # Execute all workflows in parallel
    results = await engine.execute_multiple_workflows(workflow_ids)

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    # Analyze results
    print(f"\n{'='*60}")
    print("Execution Complete")
    print(f"{'='*60}")
    print(f"Total Execution Time: {total_time:.2f}s")

    completed = sum(1 for wf in results if wf.status.value == "completed")
    print(f"\nWorkflows: {completed}/{len(results)} completed")

    total_agents = sum(len(wf.agents) for wf in results)
    completed_agents = sum(
        sum(1 for a in wf.agents if a.status.value == "completed")
        for wf in results
    )
    print(f"Agents: {completed_agents}/{total_agents} completed")

    total_cost = sum(wf.total_cost for wf in results)
    print(f"Total Cost: ${total_cost:.4f}")

    # Per-module summary
    print(f"\nPer-Module Summary:")
    for module in ["HR", "Finance", "Inventory", "Compliance"]:
        module_workflows = [wf for wf in results if wf.module == module]
        if module_workflows:
            completed = sum(1 for wf in module_workflows if wf.status.value == "completed")
            cost = sum(wf.total_cost for wf in module_workflows)
            print(f"  {module}: {completed}/{len(module_workflows)} workflows, ${cost:.4f}")

    # System metrics
    status = engine.get_system_status()
    print(f"\nSystem Metrics:")
    print(f"  API Calls: {status['rate_limiting']['total_requests']}")
    print(f"  Total Tokens: {status['cost_tracking']['total_tokens']}")
    print(f"  Avg Wait Time: {status['rate_limiting']['avg_wait_time']:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())

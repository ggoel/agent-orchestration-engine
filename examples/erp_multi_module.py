"""
Complex ERP workflow example with multiple modules and dependencies.
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


def create_hr_agents():
    """Create HR module agents."""
    return [
        Agent(
            name="resume_parser",
            module="HR",
            description="Parse and extract information from resumes",
            llm_config={"model": "gpt-4", "temperature": 0.3, "max_tokens": 1500},
            prompt_template="Parse the resume and extract key information.",
            dependencies=[],
            priority=8
        ),
        Agent(
            name="candidate_screener",
            module="HR",
            description="Screen candidates based on requirements",
            llm_config={"model": "gpt-4", "temperature": 0.4, "max_tokens": 1000},
            prompt_template="Screen the candidate based on job requirements.",
            dependencies=["resume_parser"],
            priority=7
        ),
        Agent(
            name="interview_scheduler",
            module="HR",
            description="Schedule interviews with qualified candidates",
            llm_config={"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 800},
            prompt_template="Schedule interviews based on availability.",
            dependencies=["candidate_screener"],
            priority=6
        ),
    ]


def create_finance_agents():
    """Create Finance module agents."""
    return [
        Agent(
            name="invoice_processor",
            module="Finance",
            description="Process and validate invoices",
            llm_config={"model": "gpt-4", "temperature": 0.2, "max_tokens": 1200},
            prompt_template="Process and validate the invoice data.",
            dependencies=[],
            priority=9
        ),
        Agent(
            name="payment_scheduler",
            module="Finance",
            description="Schedule payments based on terms",
            llm_config={"model": "gpt-4", "temperature": 0.3, "max_tokens": 1000},
            prompt_template="Schedule payments according to payment terms.",
            dependencies=["invoice_processor"],
            priority=8
        ),
        Agent(
            name="financial_reconciliation",
            module="Finance",
            description="Reconcile financial transactions",
            llm_config={"model": "gpt-4", "temperature": 0.2, "max_tokens": 1500},
            prompt_template="Reconcile transactions and identify discrepancies.",
            dependencies=["payment_scheduler"],
            priority=7
        ),
    ]


def create_inventory_agents():
    """Create Inventory module agents."""
    return [
        Agent(
            name="stock_analyzer",
            module="Inventory",
            description="Analyze stock levels and trends",
            llm_config={"model": "gpt-4", "temperature": 0.4, "max_tokens": 1200},
            prompt_template="Analyze stock levels and identify trends.",
            dependencies=[],
            priority=7
        ),
        Agent(
            name="reorder_optimizer",
            module="Inventory",
            description="Optimize reorder points and quantities",
            llm_config={"model": "gpt-4", "temperature": 0.5, "max_tokens": 1000},
            prompt_template="Optimize reorder points based on demand patterns.",
            dependencies=["stock_analyzer"],
            priority=6
        ),
        Agent(
            name="supplier_selector",
            module="Inventory",
            description="Select optimal suppliers",
            llm_config={"model": "gpt-4", "temperature": 0.4, "max_tokens": 1200},
            prompt_template="Select best suppliers based on cost and reliability.",
            dependencies=["reorder_optimizer"],
            priority=5
        ),
    ]


def create_compliance_agents():
    """Create Compliance module agents."""
    return [
        Agent(
            name="policy_checker",
            module="Compliance",
            description="Check compliance with policies",
            llm_config={"model": "gpt-4", "temperature": 0.2, "max_tokens": 1500},
            prompt_template="Check transactions for policy compliance.",
            dependencies=["invoice_processor", "payment_scheduler"],  # Cross-module dependency
            priority=9
        ),
        Agent(
            name="audit_report_generator",
            module="Compliance",
            description="Generate audit reports",
            llm_config={"model": "gpt-4", "temperature": 0.3, "max_tokens": 2000},
            prompt_template="Generate comprehensive audit report.",
            dependencies=["policy_checker", "financial_reconciliation"],  # Cross-module
            priority=8
        ),
    ]


async def main():
    # Initialize LLM client
    llm_client = OpenAIClient(api_key="your-api-key-here")

    # Configure engine for high-scale operations
    config = EngineConfig(
        rate_limit_config=RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=5000,
            tokens_per_minute=150000,
            concurrent_requests=20
        ),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(enable_metrics=True),
        default_retry_policy=RetryPolicy(max_retries=3),
        max_concurrent_workflows=100
    )

    # Create orchestration engine
    engine = OrchestrationEngine(llm_client=llm_client, config=config)

    # Create comprehensive workflow with all modules
    all_agents = (
        create_hr_agents() +
        create_finance_agents() +
        create_inventory_agents() +
        create_compliance_agents()
    )

    workflow = Workflow(
        name="erp_daily_operations",
        description="Daily ERP operations across all modules",
        agents=all_agents,
        max_parallel_agents=10,
        cost_budget=50.0,
        timeout=3600.0,
        allow_partial_completion=True,
        module="General"
    )

    # Register and execute
    workflow_id = await engine.register_workflow(workflow)
    print(f"Registered ERP workflow: {workflow_id}")
    print(f"Total agents: {len(all_agents)}")

    print("\nExecuting comprehensive ERP workflow...")
    result = await engine.execute_workflow(workflow_id)

    # Print detailed results
    print(f"\n{'='*60}")
    print(f"Workflow Status: {result.status.value}")
    print(f"{'='*60}")

    stats = result.get_statistics()
    print(f"\nExecution Statistics:")
    print(f"  Total Agents: {stats['total_agents']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success Rate: {stats['completed']/stats['total_agents']*100:.1f}%")
    print(f"  Total Cost: ${stats['total_cost']:.4f}")
    print(f"  Budget Used: {stats['total_cost']/stats['cost_budget']*100:.1f}%")

    # Results by module
    print(f"\nResults by Module:")
    for module in ["HR", "Finance", "Inventory", "Compliance"]:
        module_agents = [a for a in result.agents if a.module == module]
        completed = sum(1 for a in module_agents if a.status.value == "completed")
        print(f"  {module}: {completed}/{len(module_agents)} completed")

    # System status
    print(f"\n{'='*60}")
    print("System Status:")
    print(f"{'='*60}")
    status = engine.get_system_status()
    print(f"  Total Workflows: {status['engine']['total_workflows']}")
    print(f"  Total API Calls: {status['rate_limiting']['total_requests']}")
    print(f"  Total Cost: ${status['cost_tracking']['total_cost']:.4f}")
    print(f"  Avg Agent Execution: {status['monitoring']['agents']['avg_execution_time']:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

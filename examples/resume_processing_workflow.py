"""
Enhanced example: Resume processing workflow with actual data input.
Demonstrates how to pass input data (resume PDFs) to agents.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any

from src.agents.base import Agent
from src.agents.workflow import Workflow
from src.orchestrator.engine import OrchestrationEngine, EngineConfig
from src.llm_clients.openai_client import OpenAIClient
from src.rate_limiting.limiter import RateLimitConfig
from src.cost_tracking.tracker import CostConfig
from src.monitoring.monitor import MonitorConfig
from src.failure_handling.handler import RetryPolicy


# Simulated resume data - in real scenario, this would come from file system or database
RESUME_DATA = {
    "candidate_1": {
        "file_path": "/path/to/resume1.pdf",
        "text_content": """
        John Doe
        Senior Software Engineer

        Experience:
        - 5 years Python development
        - 3 years AWS cloud architecture
        - Led team of 5 engineers

        Education:
        - MS Computer Science, Stanford University
        - BS Computer Science, MIT

        Skills: Python, AWS, Docker, Kubernetes, React
        """
    },
    "candidate_2": {
        "file_path": "/path/to/resume2.pdf",
        "text_content": """
        Jane Smith
        Data Scientist

        Experience:
        - 4 years machine learning
        - 2 years data engineering
        - Published 5 papers in ML conferences

        Education:
        - PhD Machine Learning, Berkeley
        - MS Statistics, Stanford

        Skills: Python, TensorFlow, PyTorch, SQL, Spark
        """
    }
}

# Job requirements
JOB_REQUIREMENTS = {
    "position": "Senior ML Engineer",
    "required_skills": ["Python", "Machine Learning", "AWS"],
    "min_experience": 3,
    "education": "MS or PhD in Computer Science or related field"
}


def create_resume_parser_agent(candidate_id: str, resume_data: Dict[str, Any]) -> Agent:
    """
    Create resume parser agent with actual input data.

    The prompt template now includes the actual resume content.
    """
    prompt = f"""
You are a resume parsing expert. Extract structured information from the following resume.

RESUME CONTENT:
{resume_data['text_content']}

Extract and return the following information in JSON format:
- candidate_name
- contact_info
- years_of_experience
- education (list of degrees)
- skills (list)
- work_history (list of positions with company, title, duration)

Return ONLY valid JSON.
"""

    return Agent(
        name=f"resume_parser_{candidate_id}",
        module="HR",
        description=f"Parse resume for {candidate_id}",
        llm_config={
            "model": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 1500
        },
        prompt_template=prompt,
        dependencies=[],
        priority=8,
        metadata={
            "candidate_id": candidate_id,
            "file_path": resume_data['file_path']
        }
    )


def create_candidate_screener_agent(
    candidate_id: str,
    job_requirements: Dict[str, Any],
    parser_agent_name: str
) -> Agent:
    """
    Create candidate screener that uses output from parser.

    In practice, you'd use the actual output from the previous agent.
    For this example, we include job requirements in the prompt.
    """
    prompt = f"""
You are a candidate screening expert. Evaluate the candidate based on parsed resume data.

JOB REQUIREMENTS:
Position: {job_requirements['position']}
Required Skills: {', '.join(job_requirements['required_skills'])}
Minimum Experience: {job_requirements['min_experience']} years
Education: {job_requirements['education']}

Based on the parsed resume data from the previous step, provide:
1. Match score (0-100)
2. Strengths (what matches well)
3. Gaps (what's missing)
4. Recommendation (recommend, maybe, reject)

Return response in JSON format.
"""

    return Agent(
        name=f"candidate_screener_{candidate_id}",
        module="HR",
        description=f"Screen {candidate_id} against job requirements",
        llm_config={
            "model": "gpt-4",
            "temperature": 0.3,
            "max_tokens": 1000
        },
        prompt_template=prompt,
        dependencies=[parser_agent_name],  # Depends on parser
        priority=7,
        metadata={
            "candidate_id": candidate_id,
            "job_position": job_requirements['position']
        }
    )


def create_interview_scheduler_agent(
    candidate_id: str,
    screener_agent_name: str
) -> Agent:
    """
    Create interview scheduler for qualified candidates.
    """
    prompt = f"""
You are an interview scheduling assistant. Based on the candidate screening results,
if the candidate is recommended or maybe, suggest:

1. Interview type (phone screen, technical interview, or onsite)
2. Interview duration
3. Key areas to focus on
4. Interview panel suggestions (roles needed)
5. Priority level (high, medium, low)

Return response in JSON format.
"""

    return Agent(
        name=f"interview_scheduler_{candidate_id}",
        module="HR",
        description=f"Schedule interview for {candidate_id}",
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.4,
            "max_tokens": 800
        },
        prompt_template=prompt,
        dependencies=[screener_agent_name],  # Depends on screener
        priority=6,
        metadata={
            "candidate_id": candidate_id
        }
    )


async def process_candidates():
    """
    Main function to process multiple candidates.
    """
    # Initialize LLM client
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    llm_client = OpenAIClient(api_key=api_key)

    # Configure engine
    config = EngineConfig(
        rate_limit_config=RateLimitConfig(
            requests_per_minute=60,
            concurrent_requests=10
        ),
        cost_config=CostConfig(),
        monitor_config=MonitorConfig(enable_metrics=True),
        default_retry_policy=RetryPolicy(max_retries=3)
    )

    # Create engine
    engine = OrchestrationEngine(llm_client=llm_client, config=config)

    # Create agents for each candidate
    all_agents = []

    for candidate_id, resume_data in RESUME_DATA.items():
        # Create parser agent with resume data
        parser = create_resume_parser_agent(candidate_id, resume_data)

        # Create screener agent that depends on parser
        screener = create_candidate_screener_agent(
            candidate_id,
            JOB_REQUIREMENTS,
            parser.name
        )

        # Create scheduler agent that depends on screener
        scheduler = create_interview_scheduler_agent(
            candidate_id,
            screener.name
        )

        all_agents.extend([parser, screener, scheduler])

    # Create workflow
    workflow = Workflow(
        name="resume_processing_workflow",
        description="Process multiple resumes with parsing, screening, and scheduling",
        agents=all_agents,
        max_parallel_agents=6,  # Process 2 candidates in parallel (3 agents each)
        cost_budget=10.0,
        timeout=1800.0,
        allow_partial_completion=True,
        module="HR"
    )

    # Execute workflow
    print("="*70)
    print("RESUME PROCESSING WORKFLOW")
    print("="*70)
    print(f"Processing {len(RESUME_DATA)} candidates")
    print(f"Total agents: {len(all_agents)}")
    print(f"Job Position: {JOB_REQUIREMENTS['position']}")
    print("="*70)

    workflow_id = await engine.register_workflow(workflow)
    result = await engine.execute_workflow(workflow_id)

    # Print results
    print(f"\nWorkflow Status: {result.status.value}")
    print(f"Total Cost: ${result.total_cost:.4f}")
    print("="*70)

    # Print results by candidate
    for candidate_id in RESUME_DATA.keys():
        print(f"\n{'='*70}")
        print(f"CANDIDATE: {candidate_id}")
        print(f"{'='*70}")

        # Get agents for this candidate
        candidate_agents = [
            a for a in result.agents
            if a.metadata.get('candidate_id') == candidate_id
        ]

        for agent in candidate_agents:
            stage = agent.name.split('_')[0]  # resume, candidate, interview
            print(f"\n  Stage: {stage.upper()}")
            print(f"  Status: {agent.status.value}")

            if agent.result and agent.result.output:
                content = agent.result.output.get('content', '')
                # Truncate long content
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"  Output: {content}")
                print(f"  Cost: ${agent.result.cost:.4f}")
                print(f"  Time: {agent.result.execution_time:.2f}s")

    # System summary
    print(f"\n{'='*70}")
    print("SYSTEM SUMMARY")
    print(f"{'='*70}")
    status = engine.get_system_status()
    print(f"Total API Calls: {status['rate_limiting']['total_requests']}")
    print(f"Total Cost: ${status['cost_tracking']['total_cost']:.4f}")
    print(f"Avg Agent Time: {status['monitoring']['agents']['avg_execution_time']:.2f}s")


# ALTERNATIVE APPROACH: Using agent callbacks to pass data between agents
class DataStore:
    """
    Shared data store for passing data between agents.
    In production, this could be Redis, database, or object storage.
    """
    def __init__(self):
        self.data = {}

    def store(self, key: str, value: Any):
        """Store data for later retrieval."""
        self.data[key] = value

    def retrieve(self, key: str) -> Any:
        """Retrieve stored data."""
        return self.data.get(key)


async def agent_success_callback(agent: Agent, result: Any, data_store: DataStore):
    """
    Callback executed when agent completes successfully.
    Use this to extract output and make it available to dependent agents.
    """
    candidate_id = agent.metadata.get('candidate_id')

    if agent.name.startswith('resume_parser'):
        # Store parsed resume data
        data_store.store(f"parsed_resume_{candidate_id}", result.output)
        print(f"✓ Stored parsed resume for {candidate_id}")

    elif agent.name.startswith('candidate_screener'):
        # Store screening results
        data_store.store(f"screening_{candidate_id}", result.output)
        print(f"✓ Stored screening results for {candidate_id}")


async def main():
    """Main entry point."""
    await process_candidates()


if __name__ == "__main__":
    asyncio.run(main())

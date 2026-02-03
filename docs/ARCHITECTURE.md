# Architecture Documentation

## Overview

The Agent Orchestration Engine is designed as a modular, scalable system for managing complex AI agent workflows across multiple ERP modules. The architecture emphasizes:

- **Modularity**: Clear separation of concerns
- **Scalability**: Support for 500+ concurrent agents
- **Reliability**: Comprehensive failure handling
- **Observability**: Real-time monitoring and metrics
- **Cost Control**: Fine-grained budget management

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client Application                          │
│                  (Business Logic Layer)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Workflow Definition & Execution
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   Orchestration Engine                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Workflow Execution Manager                    │  │
│  │  - Workflow Registration                                 │  │
│  │  - Dependency Resolution                                 │  │
│  │  - Parallel Execution Coordination                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────┼────────────────────────────────┐ │
│  │         Agent Execution Pipeline                          │ │
│  │                                                           │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │ │
│  │  │  Rate   │→ │  Cost   │→ │   LLM   │→ │Monitor  │   │ │
│  │  │ Limiter │  │ Tracker │  │  Call   │  │         │   │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │ │
│  │                      ↓                                   │ │
│  │                  ┌─────────┐                            │ │
│  │                  │ Failure │                            │ │
│  │                  │ Handler │                            │ │
│  │                  └─────────┘                            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│   OpenAI API   │  │  Anthropic API  │  │  Custom APIs   │
└────────────────┘  └─────────────────┘  └────────────────┘
```

### Component Architecture

## 1. Orchestration Engine

**Location**: `src/orchestrator/engine.py`

**Responsibilities**:
- Workflow lifecycle management
- Agent execution coordination
- Component integration
- Concurrency control

**Key Classes**:

```python
class OrchestrationEngine:
    - llm_client: LLMClient
    - rate_limiter: RateLimiter
    - cost_tracker: CostTracker
    - failure_handler: FailureHandler
    - monitor: Monitor
    - workflows: Dict[str, Workflow]
```

**Key Methods**:
- `register_workflow(workflow)`: Register a new workflow
- `execute_workflow(workflow_id)`: Execute a single workflow
- `execute_multiple_workflows(workflow_ids)`: Execute multiple workflows in parallel
- `get_workflow_status(workflow_id)`: Get workflow status
- `get_system_status()`: Get overall system metrics

**Execution Flow**:
```
1. Workflow Registration
   └→ Validate workflow
   └→ Set budgets
   └→ Store in registry

2. Workflow Execution
   └→ Resolve dependencies (topological sort)
   └→ For each execution level:
       ├→ Check workflow budget
       ├→ Execute agents in parallel (up to max_parallel_agents)
       ├→ Handle failures
       └→ Check if workflow should continue

3. Workflow Completion
   └→ Calculate final costs
   └→ Update metrics
   └→ Return result
```

## 2. Agent System

**Location**: `src/agents/`

### Agent (`base.py`)

**Core Entity**: Represents a single AI agent

**Key Attributes**:
```python
@dataclass
class Agent:
    # Identity
    name: str
    module: str
    description: str

    # Configuration
    llm_config: Dict[str, Any]
    prompt_template: str
    tools: List[Dict[str, Any]]

    # Dependencies
    dependencies: List[str]

    # Limits
    max_retries: int
    timeout: float
    cost_limit: float
    priority: int

    # State
    status: AgentStatus
    result: Optional[AgentResult]
```

**Agent Status Lifecycle**:
```
PENDING → RUNNING → COMPLETED
              ↓
         [FAILED / RATE_LIMITED / BUDGET_EXCEEDED]
              ↓
         (Retry Logic)
              ↓
         RUNNING or CANCELLED
```

### Workflow (`workflow.py`)

**Core Entity**: Collection of agents with dependencies

**Key Features**:
- Dependency validation (cycle detection)
- Execution planning
- Statistics tracking
- Partial completion support

**Dependency Validation**:
```python
def _validate_dependencies():
    # 1. Check all dependencies exist
    # 2. Perform DFS to detect cycles
    # 3. Raise ValueError if invalid
```

## 3. Rate Limiting System

**Location**: `src/rate_limiting/limiter.py`

**Algorithm**: Token Bucket

**Multi-Tier Limiting**:
- Per-minute limits
- Per-hour limits
- Per-day limits
- Token usage limits
- Concurrent request limits

**Token Bucket Implementation**:
```
┌─────────────────────┐
│   Token Bucket      │
│  Capacity: N        │
│  Tokens: T          │
│  Refill: R/sec      │
└─────────────────────┘
        │
        ├─ consume(tokens) → bool
        │   └─ Refill based on time
        │   └─ Check if tokens >= required
        │   └─ Deduct tokens if available
        │
        └─ wait_for_tokens(tokens, timeout)
            └─ Loop until tokens available
            └─ Sleep for calculated wait time
            └─ Raise TimeoutError if exceeded
```

**Acquisition Process**:
```
1. Request to acquire(estimated_tokens)
2. Acquire concurrent semaphore
3. Wait for per-minute rate limit
4. Wait for per-hour rate limit
5. Wait for per-day rate limit
6. Wait for token usage limit
7. If any step times out → release semaphore, raise error
8. Update statistics
```

## 4. Cost Tracking System

**Location**: `src/cost_tracking/tracker.py`

**Responsibilities**:
- Calculate costs per API call
- Track costs at workflow and agent levels
- Enforce budgets
- Generate cost reports

**Cost Calculation**:
```python
cost = (input_tokens / 1000) * input_cost_per_1k +
       (output_tokens / 1000) * output_cost_per_1k
```

**Budget Enforcement**:
```
Before Agent Execution:
├─ Check workflow budget
│  └─ workflow_costs[id] < workflow_budgets[id]
└─ Check agent budget
   └─ agent_costs[id] < agent_budgets[id]

If exceeded → Set status to BUDGET_EXCEEDED
```

**Data Structures**:
```python
- entries: List[CostEntry]  # All cost records
- workflow_budgets: Dict[str, float]
- workflow_costs: Dict[str, float]
- agent_budgets: Dict[str, float]
- agent_costs: Dict[str, float]
```

## 5. Dependency Resolution

**Location**: `src/dependencies/resolver.py`

**Algorithm**: Topological Sort (Kahn's Algorithm)

**Process**:
```
1. Build dependency graph
2. Calculate in-degree for each agent
3. Start with agents having in-degree = 0
4. Process level-by-level:
   ├─ All agents in a level can execute in parallel
   ├─ After level completes, update in-degrees
   └─ Add newly available agents to next level
```

**Example**:
```
Agents: A, B, C, D, E
Dependencies:
  B depends on A
  C depends on A
  D depends on B
  E depends on C, D

Resolution:
  Level 0: [A]           (no dependencies)
  Level 1: [B, C]        (depend only on A)
  Level 2: [D]           (depends on B)
  Level 3: [E]           (depends on C, D)
```

**Critical Path**:
```python
def get_critical_path():
    # Find longest dependency chain
    # Uses dynamic programming
    # Returns list of agents in critical path
```

## 6. Failure Handling

**Location**: `src/failure_handling/handler.py`

**Retry Strategies**:

1. **Immediate**: Retry immediately
2. **Constant Delay**: Fixed delay between retries
3. **Linear Backoff**: delay = base_delay * attempt
4. **Exponential Backoff**: delay = base_delay * (base^attempt)

**Exponential Backoff with Jitter**:
```python
delay = base_delay * (exponential_base ** (attempt - 1))
delay = min(delay, max_delay)
if jitter:
    delay *= (0.5 + random() * 0.5)
```

**Failure Decision Logic**:
```
On Agent Failure:
├─ Is retriable? (FAILED or RATE_LIMITED)
│  └─ No → Permanent failure
├─ Retry count < max_retries?
│  └─ No → Permanent failure
├─ Calculate delay
├─ Wait
└─ Retry

On Workflow Failure:
├─ allow_partial_completion?
│  └─ No → Fail immediately
├─ Critical agents failed? (priority >= 8)
│  └─ Yes → Fail workflow
└─ Failure rate > 50%?
   └─ Yes → Fail workflow
```

## 7. Monitoring System

**Location**: `src/monitoring/monitor.py`

**Metrics Collected**:
```python
{
    "workflows_started": int,
    "workflows_completed": int,
    "workflows_failed": int,
    "agents_executed": int,
    "agents_failed": int,
    "total_api_calls": int,
    "total_tokens": int,
    "total_cost": float,
    "avg_agent_execution_time": float
}
```

**Events Tracked**:
- Workflow start/complete/failed
- Agent start/complete/failed
- API calls
- Budget violations
- Rate limit hits

**Event Structure**:
```python
@dataclass
class MetricEvent:
    timestamp: datetime
    event_type: str
    agent_id: Optional[str]
    workflow_id: Optional[str]
    data: Dict[str, Any]
```

## 8. LLM Client Abstraction

**Location**: `src/llm_clients/`

**Design**: Abstract base class with provider implementations

**Interface**:
```python
class LLMClient(ABC):
    @abstractmethod
    async def complete(prompt, model, ...) -> LLMResponse

    @abstractmethod
    async def complete_with_tools(messages, tools, ...) -> LLMResponse

    @abstractmethod
    def get_model_info(model) -> Dict
```

**Implementations**:
- `OpenAIClient`: OpenAI GPT models
- `AnthropicClient`: Claude models
- Custom: Easy to extend

## Data Flow

### Workflow Execution Data Flow

```
1. Client submits Workflow
   ↓
2. Engine.register_workflow()
   ├─ Validate workflow
   ├─ Set budgets in CostTracker
   └─ Store in workflows dict
   ↓
3. Engine.execute_workflow()
   ├─ Update status to RUNNING
   ├─ Record start in Monitor
   └─ Resolve dependencies
   ↓
4. For each execution level:
   ├─ Check workflow budget (CostTracker)
   ├─ Create agent execution tasks
   └─ Execute in parallel
   ↓
5. For each agent:
   ├─ Update status to RUNNING
   ├─ Record start in Monitor
   ├─ Check agent budget
   ├─ Acquire rate limit (RateLimiter)
   ├─ Call LLM (LLMClient)
   ├─ Record cost (CostTracker)
   ├─ Record metrics (Monitor)
   ├─ Handle failures (FailureHandler)
   └─ Return result
   ↓
6. Check workflow completion
   ├─ All agents completed → COMPLETED
   ├─ Some failed + allow_partial → PARTIALLY_COMPLETED
   └─ Critical failed → FAILED
   ↓
7. Finalize workflow
   ├─ Calculate total cost
   ├─ Update metrics
   └─ Return result to client
```

## Concurrency Model

### Async/Await Architecture

**Key Principles**:
- Non-blocking I/O for all network calls
- Parallel execution within limits
- Semaphore-based concurrency control

**Concurrency Layers**:

1. **Workflow Level**:
   ```python
   workflow_semaphore = Semaphore(max_concurrent_workflows)
   ```

2. **Agent Level** (within workflow):
   ```python
   max_parallel_agents per workflow
   ```

3. **API Level**:
   ```python
   concurrent_semaphore = Semaphore(concurrent_requests)
   ```

### Parallel Execution Example

```
Workflow with agents: A, B, C, D, E
Dependencies:
  B → A
  C → A
  D → B, C
  E → D

Execution Timeline:
t0: [A] starts
t1: [A] completes
t2: [B, C] start in parallel
t3: [B] completes
t4: [C] completes
t5: [D] starts
t6: [D] completes
t7: [E] starts
t8: [E] completes

Max parallelism: 2 (B and C)
Total time: ~8 time units
Sequential time would be: ~8 time units
Savings: ~20% (depends on agent durations)
```

## Scalability Considerations

### Vertical Scaling
- **Concurrency limits**: Adjust based on available resources
- **Memory**: Minimal per-agent overhead (~1KB)
- **CPU**: Primarily I/O bound, minimal CPU usage

### Horizontal Scaling
- **Multiple engine instances**: Run separate engines
- **Shared rate limiting**: Use Redis for distributed rate limiting
- **Shared monitoring**: Export metrics to centralized system (Prometheus)

### Performance Characteristics

**Single Engine Instance**:
- 500+ concurrent agents
- 100+ concurrent workflows
- 50+ API calls per second (rate limit dependent)
- <100MB memory for 500 agents

**Bottlenecks**:
1. API rate limits (primary)
2. API response latency (secondary)
3. Network bandwidth (tertiary)

## Error Handling Strategy

### Error Categories

1. **Retriable Errors**:
   - Network timeouts
   - Rate limit errors (429)
   - Temporary API errors (5xx)
   - Action: Retry with backoff

2. **Non-Retriable Errors**:
   - Invalid API key (401)
   - Invalid request (400)
   - Budget exceeded
   - Action: Fail immediately

3. **Partial Failures**:
   - Some agents succeed, some fail
   - Action: Continue if allow_partial_completion

### Error Propagation

```
Agent Error
  ↓
Failure Handler evaluates
  ↓
Retry or Fail?
  ↓
If Fail → Update agent status
  ↓
Check workflow impact
  ↓
Continue or Fail Workflow?
  ↓
Update workflow status
  ↓
Return to client
```

## Security Considerations

### API Key Management
- Never log API keys
- Use environment variables
- Rotate keys regularly

### Cost Controls
- Always set budgets
- Monitor costs in real-time
- Alert on unusual spending

### Rate Limit Protection
- Configure conservative limits
- Monitor rate limit metrics
- Handle 429 errors gracefully

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock external dependencies
- Cover edge cases

### Integration Tests
- Test component interactions
- Use test LLM endpoints
- Verify error handling

### Load Tests
- Test with 500+ agents
- Measure throughput
- Identify bottlenecks

## Future Enhancements

### Planned Features
1. **Distributed Execution**: Redis-based coordination
2. **Streaming Responses**: Support for streaming LLM calls
3. **Advanced Scheduling**: Priority queues, SLA-based scheduling
4. **Enhanced Monitoring**: Prometheus metrics, Grafana dashboards
5. **Workflow Templates**: Pre-built workflows for common tasks
6. **Agent Marketplace**: Share and reuse agents
7. **Visual Workflow Builder**: GUI for workflow creation

### Extension Points
- Custom LLM clients
- Custom rate limiters
- Custom cost calculators
- Custom failure handlers
- Custom monitoring backends

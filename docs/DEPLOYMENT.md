# Deployment Guide

Complete guide for deploying the Agent Orchestration Engine in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Monitoring](#monitoring)
6. [Scaling](#scaling)
7. [Security](#security)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum**:
- Python 3.8+
- 2 GB RAM
- 2 CPU cores
- 10 GB disk space

**Recommended for 500+ agents**:
- Python 3.10+
- 8 GB RAM
- 4 CPU cores
- 50 GB disk space (for logs and metrics)

### Dependencies

```bash
# Core
openai>=1.0.0
anthropic>=0.18.0

# Optional
redis>=5.0.0  # For distributed rate limiting
prometheus-client>=0.19.0  # For metrics
```

### External Services

- **LLM API Access**: OpenAI or Anthropic API key
- **Redis** (optional): For distributed deployments
- **Monitoring** (optional): Prometheus, Grafana

---

## Installation

### Option 1: pip install

```bash
# Clone repository
git clone https://github.com/yourorg/agent-orchestration-engine.git
cd agent-orchestration-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

### Option 2: Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY examples/ ./examples/
COPY config/ ./config/

ENV PYTHONPATH=/app

CMD ["python", "-m", "examples.basic_workflow"]
```

Build and run:
```bash
docker build -t agent-orchestration-engine .
docker run -e OPENAI_API_KEY=your-key agent-orchestration-engine
```

### Option 3: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  orchestrator:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_HOST=redis
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

volumes:
  redis_data:
  prometheus_data:
```

Run:
```bash
docker-compose up -d
```

---

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Rate Limits (adjust based on your API tier)
RATE_LIMIT_RPM=60
RATE_LIMIT_RPH=3000
RATE_LIMIT_RPD=50000
RATE_LIMIT_TPM=100000
RATE_LIMIT_CONCURRENT=10

# Cost Limits
DEFAULT_WORKFLOW_BUDGET=100.0
DEFAULT_AGENT_BUDGET=1.0

# Retry Configuration
MAX_RETRIES=3
RETRY_BASE_DELAY=1.0
RETRY_MAX_DELAY=60.0

# System Configuration
MAX_CONCURRENT_WORKFLOWS=50
LOG_LEVEL=INFO
ENABLE_METRICS=true

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Monitoring (optional)
PROMETHEUS_PORT=8000
```

### Configuration File

Create `config/production.yaml`:

```yaml
engine:
  max_concurrent_workflows: 100

rate_limiting:
  requests_per_minute: 100
  requests_per_hour: 5000
  requests_per_day: 100000
  tokens_per_minute: 150000
  concurrent_requests: 20

cost_tracking:
  model_costs:
    gpt-4:
      input: 0.03
      output: 0.06
    gpt-4-turbo:
      input: 0.01
      output: 0.03
    gpt-3.5-turbo:
      input: 0.0005
      output: 0.0015
    claude-3-opus-20240229:
      input: 0.015
      output: 0.075
    claude-3-sonnet-20240229:
      input: 0.003
      output: 0.015

retry_policy:
  max_retries: 3
  strategy: exponential_backoff
  base_delay: 1.0
  max_delay: 60.0
  jitter: true

monitoring:
  enable_metrics: true
  enable_logging: true
  metrics_interval: 60.0
```

Load configuration:

```python
import yaml
from src.orchestrator.engine import EngineConfig

with open('config/production.yaml') as f:
    config_dict = yaml.safe_load(f)

config = EngineConfig(
    rate_limit_config=RateLimitConfig(**config_dict['rate_limiting']),
    cost_config=CostConfig(**config_dict['cost_tracking']),
    # ...
)
```

---

## Deployment Options

### Option 1: Single Instance (Small Scale)

**Suitable for**: <100 concurrent workflows, <1000 agents/day

```python
# main.py
import asyncio
from src.orchestrator.engine import OrchestrationEngine
# ... configuration ...

async def main():
    engine = OrchestrationEngine(llm_client, config)

    # Run indefinitely
    while True:
        # Process workflows from queue
        await process_workflows(engine)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

Run with systemd:

```ini
# /etc/systemd/system/orchestrator.service
[Unit]
Description=Agent Orchestration Engine
After=network.target

[Service]
Type=simple
User=orchestrator
WorkingDirectory=/opt/agent-orchestration
Environment="OPENAI_API_KEY=your-key"
ExecStart=/opt/agent-orchestration/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable orchestrator
sudo systemctl start orchestrator
sudo systemctl status orchestrator
```

### Option 2: Multiple Instances (Large Scale)

**Suitable for**: 100+ concurrent workflows, 10,000+ agents/day

**Architecture**:
```
┌─────────────┐
│   Load      │
│  Balancer   │
└──────┬──────┘
       │
   ┌───┴───┬───────┬───────┐
   │       │       │       │
┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐
│Inst1│ │Inst2│ │Inst3│ │Inst4│
└──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
   │       │       │       │
   └───┬───┴───┬───┴───┬───┘
       │               │
   ┌───▼───┐       ┌───▼───┐
   │ Redis │       │ Queue │
   └───────┘       └───────┘
```

**Shared State**: Use Redis for distributed rate limiting

```python
# Distributed rate limiter (future enhancement)
from redis import Redis

redis_client = Redis(host='redis', port=6379)

# Use Redis-backed rate limiter
# (Requires implementation of distributed rate limiter)
```

### Option 3: Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator
spec:
  replicas: 4
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      containers:
      - name: orchestrator
        image: agent-orchestration-engine:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        - name: REDIS_HOST
          value: redis-service
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: orchestrator-service
spec:
  selector:
    app: orchestrator
  ports:
  - port: 8000
    targetPort: 8000
```

Deploy:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/redis.yaml
```

---

## Monitoring

### Logging

Configure structured logging:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Metrics Export

Export metrics to Prometheus:

```python
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Metrics
workflows_total = Counter('workflows_total', 'Total workflows')
workflows_failed = Counter('workflows_failed', 'Failed workflows')
agent_execution_time = Histogram('agent_execution_seconds', 'Agent execution time')
active_workflows = Gauge('active_workflows', 'Active workflows')
total_cost = Counter('total_cost_dollars', 'Total cost')

# Start metrics server
start_http_server(8000)
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'orchestrator'
    static_configs:
      - targets: ['orchestrator:8000']
```

### Grafana Dashboard

Import dashboard JSON or create panels for:
- Workflows per minute
- Success rate
- Average execution time
- Cost per hour
- Active agents
- Error rate

---

## Scaling

### Vertical Scaling

**Increase instance resources**:
- More CPU: Increase `max_concurrent_workflows`
- More memory: Support more active workflows
- Adjust rate limits based on API tier

### Horizontal Scaling

**Add more instances**:
- Use load balancer
- Share state via Redis
- Coordinate rate limits across instances

**Rate Limit Coordination**:

```python
# Distributed rate limiting with Redis
class DistributedRateLimiter:
    def __init__(self, redis_client, key_prefix):
        self.redis = redis_client
        self.prefix = key_prefix

    async def acquire(self):
        # Implement distributed token bucket using Redis
        # Use Redis INCR, EXPIRE for atomic operations
        pass
```

### Auto-Scaling

**Kubernetes HPA**:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: orchestrator-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: orchestrator
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: active_workflows
      target:
        type: AverageValue
        averageValue: "50"
```

---

## Security

### API Key Management

**Use secrets management**:

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
    --name openai-api-key \
    --secret-string "sk-..."

# Kubernetes Secrets
kubectl create secret generic api-keys \
    --from-literal=openai=sk-... \
    --from-literal=anthropic=sk-ant-...
```

**Rotate keys regularly**:
```python
def load_api_key():
    # Load from secrets manager
    # Implement key rotation logic
    pass
```

### Network Security

- Use HTTPS for all external calls
- Restrict egress to LLM APIs only
- Use VPC/private networks
- Enable firewall rules

### Budget Controls

```python
# Set strict budgets
workflow.cost_budget = 10.0  # $10 max
agent.cost_limit = 1.0  # $1 max per agent

# Monitor in real-time
if workflow.total_cost > ALERT_THRESHOLD:
    send_alert()
```

### Rate Limit Protection

```python
# Conservative rate limits
config = RateLimitConfig(
    requests_per_minute=50,  # 80% of actual limit
    requests_per_hour=2400,  # 80% of actual limit
)
```

---

## Troubleshooting

### Common Issues

#### 1. Rate Limit Errors

**Symptoms**: Many `RATE_LIMITED` agent statuses

**Solutions**:
- Reduce `requests_per_minute`
- Increase `concurrent_requests` spacing
- Check API tier limits
- Add more instances for distribution

#### 2. High Costs

**Symptoms**: Budget exceeded frequently

**Solutions**:
- Use cheaper models (gpt-3.5-turbo)
- Reduce `max_tokens` in agent configs
- Set stricter budgets
- Optimize prompts

#### 3. Slow Execution

**Symptoms**: High average execution times

**Solutions**:
- Increase `max_parallel_agents`
- Check network latency
- Optimize dependency chains
- Use faster models

#### 4. Memory Issues

**Symptoms**: OOM errors, high memory usage

**Solutions**:
- Reduce `max_concurrent_workflows`
- Limit event log size
- Clear old workflows
- Increase instance memory

#### 5. Dependency Errors

**Symptoms**: Workflows stuck, circular dependency errors

**Solutions**:
- Validate dependencies before execution
- Use `DependencyResolver` to check
- Visualize dependency graph
- Simplify dependencies

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Get detailed metrics
status = engine.get_system_status()
print(json.dumps(status, indent=2))

# Check specific workflow
workflow_status = engine.get_workflow_status(workflow_id)
print(workflow_status)

# Check recent events
events = monitor.get_recent_events(count=50, event_type="agent_failed")
for event in events:
    print(event)
```

### Health Checks

```python
# Implement health check endpoint
async def health_check():
    status = engine.get_system_status()

    # Check if system is healthy
    if status['monitoring']['workflows']['active'] > 100:
        return {"status": "unhealthy", "reason": "too many active workflows"}

    if status['rate_limiting']['avg_wait_time'] > 5.0:
        return {"status": "degraded", "reason": "high rate limit wait times"}

    return {"status": "healthy"}
```

---

## Performance Tuning

### Configuration Recommendations

**For High Throughput**:
```python
config = EngineConfig(
    max_concurrent_workflows=100,
    rate_limit_config=RateLimitConfig(
        concurrent_requests=50,
        requests_per_minute=200
    )
)

workflow = Workflow(
    max_parallel_agents=20,
    # ...
)
```

**For Cost Optimization**:
```python
# Use cheaper models
agent.llm_config = {
    "model": "gpt-3.5-turbo",
    "max_tokens": 500
}

# Strict budgets
workflow.cost_budget = 5.0
```

**For Reliability**:
```python
# Aggressive retries
policy = RetryPolicy(
    max_retries=5,
    max_delay=120.0
)

# Allow partial completion
workflow.allow_partial_completion = True
```

---

## Backup and Recovery

### Workflow State

```python
# Save workflow state
import json

def save_workflow(workflow):
    state = workflow.to_dict()
    with open(f'backup/{workflow.id}.json', 'w') as f:
        json.dump(state, f)

# Restore workflow
def restore_workflow(workflow_id):
    with open(f'backup/{workflow_id}.json') as f:
        state = json.load(f)
    # Reconstruct workflow
```

### Metrics Backup

```python
# Export metrics regularly
metrics = monitor.get_metrics()
with open(f'metrics/{timestamp}.json', 'w') as f:
    json.dump(metrics, f)
```

---

## Production Checklist

- [ ] API keys configured and secured
- [ ] Rate limits set appropriately
- [ ] Budgets configured
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Health checks implemented
- [ ] Auto-scaling configured (if applicable)
- [ ] Alerts configured
- [ ] Backup strategy defined
- [ ] Security review completed
- [ ] Load testing performed
- [ ] Documentation updated
- [ ] On-call rotation defined
- [ ] Incident response plan created

---

## Support

For deployment issues:
- Check logs: `tail -f logs/orchestrator.log`
- Review metrics: `http://localhost:9090` (Prometheus)
- GitHub Issues: [github.com/yourorg/agent-orchestration-engine/issues](https://github.com/yourorg/agent-orchestration-engine/issues)

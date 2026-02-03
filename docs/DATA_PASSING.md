# Data Passing Between Agents

Guide for passing input data and sharing outputs between agents in workflows.

## Problem

Agents need to:
1. Receive input data (e.g., resume PDFs, invoices, documents)
2. Share outputs with dependent agents
3. Access shared context and state

## Solutions

### Approach 1: Embed Data in Prompt Template (Recommended for Simple Cases)

**Best for**: Small text data, configuration, parameters

```python
# Get input data
resume_text = load_resume_from_file("resume.pdf")

# Create agent with data embedded in prompt
agent = Agent(
    name="resume_parser",
    module="HR",
    description="Parse resume",
    llm_config={"model": "gpt-4"},
    prompt_template=f"""
    Parse the following resume and extract key information:

    RESUME:
    {resume_text}

    Extract: name, email, skills, experience
    """
)
```

**Pros**:
- Simple and straightforward
- No external dependencies
- Works well for text data

**Cons**:
- Limited by token limits
- Not suitable for large files
- Data embedded at agent creation time

### Approach 2: Use Agent Metadata for Context

**Best for**: File paths, IDs, references

```python
agent = Agent(
    name="resume_parser",
    module="HR",
    description="Parse resume",
    llm_config={"model": "gpt-4"},
    prompt_template="Parse the resume at the specified path",
    metadata={
        "file_path": "/path/to/resume.pdf",
        "candidate_id": "12345",
        "job_id": "67890"
    }
)

# Later, access metadata in callbacks
def on_success(agent, result):
    file_path = agent.metadata["file_path"]
    candidate_id = agent.metadata["candidate_id"]
    # Process...
```

**Pros**:
- Keeps data separate from prompt
- Good for references and IDs
- Accessible throughout workflow

**Cons**:
- Still need to load actual data somewhere

### Approach 3: Shared Data Store (Recommended for Production)

**Best for**: Large data, multiple agents, complex workflows

```python
from typing import Any, Dict

class DataStore:
    """Shared data store for agent workflows."""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def store(self, key: str, value: Any):
        """Store data."""
        self._data[key] = value

    def retrieve(self, key: str) -> Any:
        """Retrieve data."""
        return self._data.get(key)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data


# Usage
data_store = DataStore()

# Store input data
data_store.store("candidate_123_resume", resume_content)
data_store.store("job_456_requirements", job_requirements)

# Agent can reference data store keys
agent1 = Agent(
    name="resume_parser",
    module="HR",
    description="Parse resume from data store",
    llm_config={"model": "gpt-4"},
    prompt_template="Parse resume",
    metadata={
        "data_key": "candidate_123_resume"
    }
)

# Callback to store results
async def on_agent_success(agent, result):
    data_key = agent.metadata.get("data_key")
    output_key = f"{data_key}_parsed"
    data_store.store(output_key, result.output)
```

**Pros**:
- Scalable for large data
- Easy to share between agents
- Can use external stores (Redis, S3, database)
- Supports caching

**Cons**:
- Requires setup
- Need to manage keys

### Approach 4: Extended Agent Class with Data Loading

**Best for**: Custom workflows with specific data needs

```python
from src.agents.base import Agent
from typing import Optional

class DataAwareAgent(Agent):
    """Agent that can load its own input data."""

    def __init__(self, *args, data_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self._input_data = None

    async def load_data(self):
        """Load input data before execution."""
        if self.data_loader:
            self._input_data = await self.data_loader(self.metadata)
            # Update prompt with loaded data
            self.prompt_template = self.prompt_template.format(
                data=self._input_data
            )

    def get_input_data(self) -> Optional[Any]:
        """Get loaded input data."""
        return self._input_data


# Data loader function
async def load_resume(metadata):
    file_path = metadata.get("file_path")
    # Load from file, S3, database, etc.
    with open(file_path) as f:
        return f.read()

# Create agent with data loader
agent = DataAwareAgent(
    name="resume_parser",
    module="HR",
    description="Parse resume",
    llm_config={"model": "gpt-4"},
    prompt_template="Parse this resume:\n{data}",
    data_loader=load_resume,
    metadata={"file_path": "/path/to/resume.pdf"}
)

# Load data before execution
await agent.load_data()
```

### Approach 5: Tool Use for Data Access

**Best for**: Dynamic data fetching during agent execution

```python
# Define tool for data access
tools = [
    {
        "type": "function",
        "function": {
            "name": "load_resume",
            "description": "Load resume content from file system",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "description": "Candidate ID"
                    }
                },
                "required": ["candidate_id"]
            }
        }
    }
]

agent = Agent(
    name="resume_parser",
    module="HR",
    description="Parse resume using data loading tool",
    llm_config={"model": "gpt-4"},
    prompt_template="Load and parse resume for candidate ID: 12345",
    tools=tools
)

# Implement tool execution
def execute_tool(tool_name, arguments):
    if tool_name == "load_resume":
        candidate_id = arguments["candidate_id"]
        return load_resume_from_storage(candidate_id)
```

## Complete Example: Resume Processing Pipeline

```python
import asyncio
from typing import Dict, Any

class ResumeProcessor:
    """Complete resume processing with data passing."""

    def __init__(self, engine):
        self.engine = engine
        self.data_store = DataStore()

    async def process_resume(self, candidate_id: str, resume_path: str):
        """Process a single resume through the pipeline."""

        # 1. Load resume content
        resume_content = self._load_resume(resume_path)
        self.data_store.store(f"resume_{candidate_id}", resume_content)

        # 2. Create parser agent
        parser = Agent(
            name=f"parser_{candidate_id}",
            module="HR",
            description="Parse resume",
            llm_config={"model": "gpt-4"},
            prompt_template=f"Parse this resume:\n\n{resume_content}",
            metadata={"candidate_id": candidate_id}
        )
        parser.on_success = lambda a, r: self._store_parsed_data(
            candidate_id, r.output
        )

        # 3. Create screener agent (depends on parser)
        screener = Agent(
            name=f"screener_{candidate_id}",
            module="HR",
            description="Screen candidate",
            llm_config={"model": "gpt-4"},
            prompt_template=self._build_screener_prompt(candidate_id),
            dependencies=[parser.name],
            metadata={"candidate_id": candidate_id}
        )
        screener.on_success = lambda a, r: self._store_screening(
            candidate_id, r.output
        )

        # 4. Create scheduler agent (depends on screener)
        scheduler = Agent(
            name=f"scheduler_{candidate_id}",
            module="HR",
            description="Schedule interview",
            llm_config={"model": "gpt-3.5-turbo"},
            prompt_template=self._build_scheduler_prompt(candidate_id),
            dependencies=[screener.name],
            metadata={"candidate_id": candidate_id}
        )

        # 5. Create and execute workflow
        workflow = Workflow(
            name=f"resume_workflow_{candidate_id}",
            description="Resume processing pipeline",
            agents=[parser, screener, scheduler],
            cost_budget=5.0
        )

        wf_id = await self.engine.register_workflow(workflow)
        result = await self.engine.execute_workflow(wf_id)

        return result

    def _load_resume(self, path: str) -> str:
        """Load resume from file."""
        # In production: handle PDFs, DOCX, etc.
        with open(path) as f:
            return f.read()

    async def _store_parsed_data(self, candidate_id: str, output: Dict):
        """Store parsed resume data."""
        self.data_store.store(f"parsed_{candidate_id}", output)

    async def _store_screening(self, candidate_id: str, output: Dict):
        """Store screening results."""
        self.data_store.store(f"screening_{candidate_id}", output)

    def _build_screener_prompt(self, candidate_id: str) -> str:
        """Build screener prompt with parsed data."""
        parsed = self.data_store.retrieve(f"parsed_{candidate_id}")
        return f"Screen this candidate:\n{parsed}"

    def _build_scheduler_prompt(self, candidate_id: str) -> str:
        """Build scheduler prompt with screening results."""
        screening = self.data_store.retrieve(f"screening_{candidate_id}")
        return f"Schedule interview based on:\n{screening}"
```

## Best Practices

### 1. Use Appropriate Approach for Data Size

- **Small text (<2KB)**: Embed in prompt
- **Medium data (2-10KB)**: Use metadata + data store
- **Large data (>10KB)**: Use data store + references
- **Files (PDFs, images)**: Extract text/features first, then process

### 2. Handle File Types Properly

```python
import PyPDF2
from PIL import Image
import pytesseract

def load_resume(file_path: str) -> str:
    """Load resume handling multiple formats."""
    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        with open(file_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            text = '\n'.join(page.extract_text() for page in pdf.pages)
        return text

    elif ext in ['.jpg', '.png']:
        # OCR for images
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text

    elif ext == '.txt':
        with open(file_path) as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")
```

### 3. Cache Expensive Operations

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def load_and_parse_resume(file_path: str) -> Dict:
    """Load and parse resume with caching."""
    content = load_resume(file_path)
    # Expensive parsing...
    return parsed_data
```

### 4. Use Async for I/O Operations

```python
import aiofiles

async def load_resume_async(file_path: str) -> str:
    """Load resume asynchronously."""
    async with aiofiles.open(file_path, 'r') as f:
        return await f.read()
```

### 5. Handle Large Datasets

```python
async def process_many_resumes(resume_paths: list):
    """Process many resumes efficiently."""

    # Batch processing
    batch_size = 10
    for i in range(0, len(resume_paths), batch_size):
        batch = resume_paths[i:i+batch_size]

        # Create agents for batch
        agents = []
        for path in batch:
            agent = create_resume_agent(path)
            agents.append(agent)

        # Process batch
        workflow = Workflow(
            name=f"batch_{i}",
            agents=agents,
            max_parallel_agents=10
        )
        await engine.execute_workflow(workflow.id)
```

## Production Considerations

### 1. External Data Stores

```python
import redis
import boto3

class RedisDataStore(DataStore):
    """Redis-backed data store."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    def store(self, key: str, value: Any):
        self.redis.set(key, json.dumps(value))

    def retrieve(self, key: str) -> Any:
        data = self.redis.get(key)
        return json.loads(data) if data else None


class S3DataStore(DataStore):
    """S3-backed data store for large files."""

    def __init__(self, bucket: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket

    def store(self, key: str, value: Any):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(value)
        )

    def retrieve(self, key: str) -> Any:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj['Body'].read())
```

### 2. Error Handling

```python
async def safe_load_data(agent: Agent) -> Optional[str]:
    """Safely load data with error handling."""
    try:
        file_path = agent.metadata.get("file_path")
        if not file_path:
            raise ValueError("No file path in metadata")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return await load_resume_async(file_path)

    except Exception as e:
        print(f"Error loading data for {agent.name}: {e}")
        return None
```

## Summary

Choose the right approach based on your needs:

1. **Simple text data**: Embed in prompt template
2. **File references**: Use metadata
3. **Complex workflows**: Use DataStore
4. **Dynamic data**: Use tool functions
5. **Custom needs**: Extend Agent class

For the resume example specifically, use **Approach 3 (Shared Data Store)** for production systems.

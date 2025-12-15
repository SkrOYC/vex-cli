# 01 - Dependencies & Compatibility

## Overview

This document outlines the dependency changes required to adopt DeepAgents as the core engine while maintaining compatibility with existing Mistral Vibe functionality.

## New Dependencies

### Primary Dependencies

```toml
[project.dependencies]
deepagents = ">=0.3.0"  # Core agent engine
langchain = "^0.3.0"   # Required by DeepAgents
langgraph = "^0.2.0"   # Graph-based agent execution
langchain-anthropic = "^0.3.0"  # For Anthropic models (optional)
langchain-mistralai = "^0.2.0"  # For Mistral models
```

### DeepAgents Transitive Dependencies

These come automatically with DeepAgents:
- `langchain-core` - Base abstractions
- `langgraph-checkpoint` - State persistence
- `pydantic` - Already used by Mistral Vibe

## Dependencies to Remove

These become redundant with DeepAgents adoption:

```toml
# No longer needed - handled by DeepAgents/LangChain
# mistralai = "^1.0.0"  # Direct SDK → use langchain-mistralai instead
```

## Compatibility Matrix

| Component | Current | After Migration | Notes |
|-----------|---------|-----------------|-------|
| Python | 3.12+ | 3.12+ | No change |
| Pydantic | v2 | v2 | Compatible |
| httpx | Used | Used by LangChain | Compatible |
| Textual | Used | Preserved | No change |

## Integration Points

### LangChain Model Integration

```python
# Before: Custom MistralBackend
from vibe.core.llm.backend.mistral import MistralBackend

# After: LangChain ChatMistralAI
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="devstral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)
```

### DeepAgents Import Structure

```python
# Core imports
from deepagents import create_deep_agent
from deepagents.middleware import FilesystemMiddleware, SubAgentMiddleware
from deepagents.backends import FilesystemBackend

# LangChain middleware (comes with DeepAgents)
from langchain.agents.middleware import (
    TodoListMiddleware,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
```

## Environment Variables

### Existing (Preserved)

```bash
MISTRAL_API_KEY=your_key_here
```

### New (Optional)

```bash
# For Anthropic models (optional)
ANTHROPIC_API_KEY=your_key_here

# For web search capabilities (optional)
TAVILY_API_KEY=your_key_here
```

## pyproject.toml Changes

```toml
[project]
name = "mistral-vibe"
version = "2.0.0"  # Major version bump for DeepAgents migration
requires-python = ">=3.12"

dependencies = [
    # Core - DeepAgents
    "deepagents>=0.1.0",
    "langchain-mistralai>=0.2.0",
    
    # TUI - Preserved
    "textual>=0.89.0",
    "rich>=13.0.0",
    
    # Config - Preserved
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "tomli>=2.0.0",
    "tomli-w>=1.0.0",
    
    # Utilities - Preserved
    "aiofiles>=24.0.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
anthropic = ["langchain-anthropic>=0.3.0"]
```

## Migration Steps

### Step 1: Add DeepAgents

```bash
uv add deepagents langchain-mistralai
```

### Step 2: Verify Compatibility

```bash
uv run python -c "from deepagents import create_deep_agent; print('OK')"
uv run python -c "from langchain_mistralai import ChatMistralAI; print('OK')"
```

### Step 3: Run Existing Tests

```bash
uv run pytest tests/ -v
```

### Step 4: Remove Deprecated Dependencies

```bash
# After full migration is complete
uv remove mistralai  # If directly used
```

## Potential Conflicts

### Pydantic Version

Both Mistral Vibe and DeepAgents use Pydantic v2. No conflicts expected.

### httpx Version

LangChain uses httpx internally. Version should align with existing usage.

### asyncio Patterns

DeepAgents uses standard asyncio patterns. Compatible with Textual's async model.

## Validation Checklist

- [ ] DeepAgents imports successfully
- [ ] LangChain Mistral integration works
- [ ] Existing Textual UI still renders
- [ ] Configuration loading unchanged
- [ ] No dependency conflicts in lock file
- [ ] All existing tests pass

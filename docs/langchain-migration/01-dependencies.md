# 01 - Dependencies & Compatibility

## Overview

This document outlines the dependency changes required to adopt LangChain 1.2.0 as the core engine, replacing DeepAgents entirely while maintaining compatibility with existing Mistral Vibe functionality.

## New Dependencies

### Primary Dependencies

```toml
[project.dependencies]
langchain = "^1.2.0"            # Core agent engine (v1.2.0, NOT 0.3.x!)
langgraph = ">=1.0.2,<1.1.0"    # Graph-based agent execution (REQUIRED by LangChain 1.2.0)
langchain-core = ">=1.2.1,<2.0.0"  # Core abstractions (transitive)
pydantic = ">=2.7.4,<3.0.0"     # Data validation (transitive)
```

### Integration-Specific Dependencies

```toml
# Add based on model providers used
# Note: Version numbers verified for LangChain 1.2.0 compatibility
langchain-openai = "^1.1.6"        # For OpenAI models (optional)
langchain-anthropic = "^1.3.1"     # For Anthropic models (optional)
langchain-mistralai = "^1.1.1"     # For Mistral models
langchain-mcp-adapters = "^0.2.1"  # MCP integration (Model Context Protocol)
```

### Verified Provider Versions

| Package | Correct Version | Verification |
|---------|----------------|--------------|
| `langchain-openai` | ^1.1.6 | Requires `langchain-core>=1.2.2`, compatible with 1.2.0+ |
| `langchain-anthropic` | ^1.3.1 | Requires `langchain-core>=1.2.0`, compatible with 1.2.0+ |
| `langchain-mistralai` | ^1.1.1 | Requires `langchain-core>=1.1.0`, compatible with 1.2.0+ |
| `langchain-mcp-adapters` | ^0.2.1 | Requires `langchain-core>=1.0.0`, compatible with 1.2.0+ |

### DeepAgents Transitive Dependencies (To Remove)

These come automatically with DeepAgents and must be removed when DeepAgents is removed:

- `langchain-core` - Now direct dependency
- `langgraph` - Now direct dependency with version constraint
- `pydantic` - Already used by Mistral Vibe

## Dependencies to Remove

```toml
# REMOVE: DeepAgents and its transitive dependencies
deepagents = ">=0.3.0"  # Full removal - no longer needed

# Already covered by LangChain 1.2.0:
# langchain-core comes transitively
# langgraph comes transitively
```

## LangChain 1.2.0 vs 0.3.x Key Differences

| Aspect | LangChain 0.3.x (Classic) | LangChain 1.2.0 |
|--------|---------------------------|-----------------|
| **Agent Creation** | Multiple functions (`create_openai_functions_agent`, etc.) | Single `create_agent()` |
| **Middleware** | No middleware concept | Native `AgentMiddleware` class |
| **State Management** | Chain-based, `ConversationBufferMemory` | TypedDict-based `AgentState` with checkpointers |
| **Streaming** | Callback-based via `AgentExecutor` | Native graph streaming with modes |
| **Package Structure** | Monolithic | Modular (core + extras) |
| **LangGraph** | Optional | Required dependency |

### Why These Changes?

LangChain 1.2.0 represents a major architectural shift from chain-based to graph-based agents, offering:

1. **Better Reliability**: LangGraph's durable execution handles interruptions and retries
2. **Structured Output**: Native support for Pydantic-based responses
3. **Middleware Extensibility**: Composable hooks for customization
4. **Unified API**: Single `create_agent()` replaces fragmented factory functions
5. **Performance**: Optimized for production use cases

## Version Compatibility Matrix

| Component | Current | After Migration | Notes |
|-----------|---------|-----------------|-------|
| **Python** | 3.12+ | 3.12+ | No change |
| **LangChain** | None (using DeepAgents) | ^1.2.0 | Major version upgrade |
| **LangGraph** | ^0.2.0 (via DeepAgents) | >=1.0.2,<1.1.0 | Updated to 1.0.x |
| **Pydantic** | v2 | v2 | Compatible, ^2.7.4 required |
| **Textual** | Used | Used | No change |
| **httpx** | Used | Used by LangChain | Compatible |

## Integration Points

### Before: DeepAgents Integration

```python
# Old: vibe/core/engine/engine.py
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware, SubAgentMiddleware
```

### After: LangChain 1.2.0 Integration

```python
# New: vibe/core/engine/langchain_engine.py
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langgraph.checkpoint.memory import InMemorySaver
```

### Model Integration

```python
# Before: Custom backend
from vibe.core.llm.backend.mistral import MistralBackend

# After: LangChain ChatMistralAI (already in use via create_model_from_config)
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="devstral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)
```

## Environment Variables

### Existing (Preserved)

```bash
MISTRAL_API_KEY=your_key_here
```

### New (Optional)

```bash
# Already supported via LangChain integrations
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## pyproject.toml Changes

### Current (with DeepAgents)

```toml
[project.dependencies]
deepagents = ">=0.3.0"
langchain = "^0.3.0"   # Via DeepAgents
langgraph = "^0.2.0"   # Via DeepAgents
```

### After Migration (remove DeepAgents, update LangChain)

```toml
[project.dependencies]
langchain = "^1.0.0"           # Core - NEW!
langgraph = ">=1.0.2,<1.1.0"   # Required by LangChain 1.2.0
langchain-mistralai = "^0.2.0" # For Mistral models
langchain-mcp-adapters = "^0.1.0"  # For MCP tools
pydantic = ">=2.7.4,<3.0.0"    # Explicit (already transitive)

[project.optional-dependencies]
anthropic = ["langchain-anthropic"]  # Add if using Anthropic
openai = ["langchain-openai"]        # Add if using OpenAI
```

## Migration Checklist

- [ ] Remove `deepagents` from dependencies
- [ ] Update `langchain` to `^1.0.0`
- [ ] Add explicit `langgraph = ">=1.0.2,<1.1.0"` dependency
- [ ] Verify `pydantic >=2.7.4` constraint
- [ ] Update any imports from `langchain-classic` patterns
- [ ] Test model provider integrations
- [ ] Verify MCP adapter compatibility
- [ ] Run existing test suite
- [ ] Ensure no DeepAgents imports remain

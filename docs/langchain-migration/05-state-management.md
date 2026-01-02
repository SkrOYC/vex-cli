# 05 - State Management

## Overview

Implement proper state management using LangChain 1.2.0's TypedDict-based `AgentState` pattern with checkpointing, replacing manual token estimation and state tracking.

## Current State Management

```python
# vibe/core/engine/engine.py

class VibeEngineStats:
    def __init__(self, messages: int = 0, context_tokens: int = 0, todos=None):
        self._messages = messages
        self._todos = todos or []
        self.steps = 0
        self.session_prompt_tokens = context_tokens // 2  # Rough estimate!
        self.session_completion_tokens = context_tokens // 2  # Rough estimate!

def _estimate_tokens(self, messages: list) -> int:
    """Estimate token count from messages using actual token metadata if available."""
    total_tokens = 0
    for msg in messages:
        # Try usage metadata first
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            total_tokens += msg.usage_metadata.get("input_tokens", 0)
            total_tokens += msg.usage_metadata.get("output_tokens", 0)
        else:
            # Fall back to character-based estimation
            total_tokens += len(str(msg.content)) // 4
```

## New State Management

```python
# New: vibe/core/engine/state.py

from typing import Annotated, TypedDict
from operator import add
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Custom state schema for the Vibe agent.
    
    LangChain 1.2.0 uses TypedDict for state schemas with reducers.
    """
    
    # Chat messages with automatic deduplication and ordering
    # Uses add_messages reducer for intelligent merging
    messages: Annotated[list, add_messages]
    
    # Token count from actual usage_metadata (no estimation!)
    context_tokens: int
    
    # Warning messages from ContextWarningMiddleware
    warning: str | None


class CheckpointConfig(TypedDict):
    """Configuration for checkpointing."""
    
    type: str  # "memory" for now, user's SQLite later
    path: str | None  # For user's custom implementation
```

## State Usage in Engine

```python
# New: vibe/core/engine/langchain_engine.py (partial)

async def run(self, user_message: str) -> AsyncGenerator[Any, None]:
    """Run a conversation turn, yielding native LangGraph events."""
    if self._agent is None:
        self.initialize()

    config: RunnableConfig = {
        "configurable": {"thread_id": self._thread_id},
        "recursion_limit": self.config.max_recursion_depth,
    }

    # Native state access - no estimation needed!
    state = self._agent.get_state(config)
    messages = state.values.get("messages", [])
    context_tokens = state.values.get("context_tokens", 0)
    warning = state.values.get("warning")

    # Stream native events
    async for event in self._agent.astream_events(
        {"messages": [("user", user_message)]},
        config=config,
        version="v2",
    ):
        yield event

@property
def stats(self) -> "VibeEngineStats":
    """Get current session statistics from native state."""
    if self._agent is not None:
        state = self._agent.get_state(
            {"configurable": {"thread_id": self._thread_id}}
        )
        
        messages = state.values.get("messages", [])
        
        # ACTUAL token counts from usage_metadata (no estimation!)
        context_tokens = 0
        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                context_tokens += msg.usage_metadata.get("input_tokens", 0)
                context_tokens += msg.usage_metadata.get("output_tokens", 0)
        
        self._stats.context_tokens = context_tokens
        self._stats._messages = len(messages)
        self._stats._todos = state.values.get("todos", [])

    return self._stats
```

## Checkpointing

```python
# New: vibe/core/engine/checkpoint.py

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver


class CheckpointManager:
    """Manage checkpoints for conversation persistence."""
    
    def __init__(self, checkpoint_type: str = "memory"):
        self.checkpointer = self._create_checkpointer(checkpoint_type)
    
    def _create_checkpointer(self, checkpoint_type: str) -> BaseCheckpointSaver:
        """Create the appropriate checkpointer."""
        match checkpoint_type:
            case "memory":
                return InMemorySaver()
            case "sqlite":
                # Placeholder for user's custom SQLite implementation
                from vibe.core.engine.checkpoint_sqlite import SqliteSaver
                return SqliteSaver()
            case _:
                raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
    
    def get_state(self, thread_id: str):
        """Get state for a thread."""
        return self.checkpointer.get(
            {"configurable": {"thread_id": thread_id}}
        )
    
    def list_threads(self):
        """List all threads."""
        return self.checkpointer.list()
```

## State Comparison

| Aspect | Before (DeepAgents) | After (LangChain 1.2.0) |
|--------|---------------------|--------------------------|
| **State Schema** | Implicit | Explicit `TypedDict` |
| **Token Tracking** | Manual `_estimate_tokens()` | Native `usage_metadata` |
| **Message Handling** | Custom list management | `add_messages` reducer |
| **Checkpointing** | Basic `InMemorySaver` | Full `BaseCheckpointSaver` |
| **State Access** | Via `get_state()` | Via `get_state()` (same) |
| **Persistence** | In-memory only | Ready for custom SQLite |

## Validation Checklist

- [ ] State schema is properly defined
- [ ] Token counts are accurate (no estimation)
- [ ] Messages are properly managed
- [ ] Checkpoints save and restore correctly
- [ ] Stats are derived from native state
- [ ] Ready for user's custom SQLite checkpointing

## ⚠️ Critical Issues Found (Post-Migration Audit)

1. **VibeAgentState Missing Annotations** ⚠️ MEDIUM
   - Missing `NotRequired`, `EphemeralValue`, `PrivateStateAttr`
   - Breaks LangGraph state management
   - **Fix Required**: Add proper LangGraph state annotations

## ⚠️ Critical Issues Found (Post-Migration Audit)

1. **VibeAgentState Missing LangGraph Type Annotations** ⚠️ MEDIUM
   - Missing `NotRequired`, `EphemeralValue`, `PrivateStateAttr`, `OmitFromInput`, `OmitFromOutput`
   - Breaks LangGraph state management behavior:
     - `warning` field persists incorrectly across turns (should be ephemeral)
     - `context_tokens` has no default value (causes errors)
     - Internal fields exposed in input/output schemas
   - **Fix Required:** Add proper LangGraph annotations (see Priority 1 in 00-overview.md)

2. **Manual Token Counting in Engine** ⚠️ LOW
   - `VibeLangChainEngine._get_actual_token_count()` sums usage_metadata manually
   - Should prefer cumulative tracking from state or middleware
   - **Fix Required:** Migrate to state-based tracking (see Priority 2 in 00-overview.md)

**Impact:**
- State schema doesn't integrate properly with LangGraph's state management
- Warnings persist incorrectly across graph steps
- May cause unexpected behavior in state persistence

**Required Fixes:**
1. Add TYPE_CHECKING imports for LangGraph channel annotations
2. Annotate `warning` with `EphemeralValue`, `PrivateStateAttr`, `OmitFromInput`, `OmitFromOutput`
3. Annotate `context_tokens` with `NotRequired`, `PrivateStateAttr`
4. Update ContextWarningMiddleware to use state's `warning` field correctly

2. **Manual Token Counting in Stats** ⚠️ LOW
   - Engine uses manual `_get_actual_token_count()` instead of `usage_metadata`
   - **Fix Required**: Migrate to state-based tracking

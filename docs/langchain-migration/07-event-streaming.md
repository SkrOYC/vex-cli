# 07 - Event Streaming

## Overview

Replace the `EventTranslator` adapter with native LangGraph streaming events, enabling direct access to all streaming modes without translation overhead.

## Current Event System

### `vibe/core/engine/adapters.py` (EventTranslator - partial)

```python
class EventTranslator:
    """Translate LangGraph events to Vibe TUI events."""
    
    def translate(self, event: dict[str, Any] | Any) -> BaseEvent | None:
        """Translate a single LangGraph event to Vibe TUI events."""
        event_type = event_data.get("event")
        
        match event_type:
            case "on_chat_model_stream":
                # Streaming token
                chunk = event_data.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    return AssistantEvent(content=chunk.content)
            
            case "interrupt":
                # Agent interrupt requiring approval
                return InterruptEvent(interrupt_data=event_data)
            
            case "on_tool_start":
                # Tool call starting
                return ToolCallEvent(...)
            
            case "on_tool_end":
                # Tool call complete
                return ToolResultEvent(...)
        
        return None
```

## New Event System

### Native LangGraph Streaming

```python
# New: vibe/core/engine/langchain_engine.py (partial)

async def run(self, user_message: str) -> AsyncGenerator[Any, None]:
    """Run a conversation turn, yielding native LangGraph events.
    
    LangChain 1.2.0 provides native streaming with multiple modes:
    - "values": Full state after each node
    - "updates": State deltas per node  
    - "messages": Token-by-token LLM output
    - "custom": User-defined events
    """
    if self._agent is None:
        self.initialize()

    config: RunnableConfig = {
        "configurable": {"thread_id": self._thread_id},
        "recursion_limit": self.config.max_recursion_depth,
    }

    # Stream native LangGraph events - NO EventTranslator!
    async for event in self._agent.astream_events(
        {"messages": [("user", user_message)]},
        config=config,
        version="v2",
    ):
        # Direct event access - can be used as-is or mapped to Vibe events
        yield event
```

### TUI Event Mapping (Optional)

```python
# New: vibe/core/engine/tui_events.py

"""Optional: Map native LangGraph events to Vibe TUI events if needed."""

from vibe.core.types import (
    AssistantEvent,
    InterruptEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class TUIEventMapper:
    """Map native LangGraph events to Vibe TUI events if needed.
    
    Note: This is OPTIONAL. The TUI can consume native events directly
    if updated to handle the LangGraph event format.
    """
    
    @staticmethod
    def to_tui_event(event: dict) -> BaseEvent | None:
        """Map a LangGraph event to a Vibe TUI event."""
        event_type = event.get("event")
        
        match event_type:
            case "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    return AssistantEvent(content=chunk.content)
            
            case "interrupt":
                return InterruptEvent(interrupt_data=event)
            
            case "on_tool_start":
                return TUIEventMapper._map_tool_start(event)
            
            case "on_tool_end":
                return TUIEventMapper._map_tool_end(event)
        
        return None
```

## Streaming Modes

```python
# LangChain 1.2.0 supports multiple streaming modes

async for event in self._agent.astream_events(
    {"messages": [("user", user_message)]},
    config=config,
    version="v2",
):
    # Default: All events from all nodes
    yield event

# Alternative: Stream only state values
async for state in self._agent.astream(
    {"messages": [("user", user_message)]},
    config=config,
    stream_mode="values",
):
    yield state

# Alternative: Stream only updates (deltas)
async for update in self._agent.astream(
    {"messages": [("user", user_message)]},
    config=config,
    stream_mode="updates",
):
    yield update
```

## Comparison

| Aspect | Before (DeepAgents) | After (LangChain 1.2.0) |
|--------|---------------------|--------------------------|
| **Event Source** | Translated via `EventTranslator` | Native from `astream_events()` |
| **Streaming Modes** | Single mode | Multiple modes (`values`, `updates`, `messages`, `custom`) |
| **Event Types** | Vibe-specific | Native LangGraph events |
| **Token Events** | Via `AssistantEvent` | Via `on_chat_model_stream` |
| **Tool Events** | Via `ToolCallEvent`/`ToolResultEvent` | Via `on_tool_start`/`on_tool_end` |
| **Interrupt Events** | Via `InterruptEvent` | Via `interrupt` event |
| **Code LOC** | ~100 lines in adapter | ~0 lines (native) |

## Validation Checklist

- [ ] Native events stream correctly
- [ ] Token-by-token streaming works
- [ ] Tool start/end events are captured
- [ ] Interrupt events are detected
- [ ] No EventTranslator code remains
- [ ] TUI can consume native events (or mapping works)

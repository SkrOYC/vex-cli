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

### Native LangGraph Streaming with TUIEventMapper

```python
# New: vibe/core/engine/langchain_engine.py

class VibeLangChainEngine:
    """LangChain 1.2.0 powered agent engine for Mistral Vibe."""
    
    def __init__(self, config: VibeConfig):
        self.config = config
        self._tui_event_mapper: TUIEventMapper | None = None
    
    def _get_tui_event_mapper(self) -> TUIEventMapper:
        """Lazy initialization of TUI event mapper."""
        if self._tui_event_mapper is None:
            self._tui_event_mapper = TUIEventMapper(self.config)
        return self._tui_event_mapper

    async def run(self, user_message: str) -> AsyncGenerator[Any, None]:
        """Run a conversation turn, yielding mapped Vibe TUI events.
        
        This method streams native LangGraph events through TUIEventMapper
        to convert them to Vibe TUI event types that EventHandler expects.
        """
        if self._agent is None:
            self.initialize()

        config: RunnableConfig = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": self.config.max_recursion_depth,
        }

        messages = [("user", user_message)]
        mapper = self._get_tui_event_mapper()

        # Stream native LangGraph events and map to Vibe TUI events
        async for event in self._agent.astream_events(
            {"messages": messages},
            config=config,
            version="v2",
        ):
            mapped_event = mapper.map_event(event)
            if mapped_event is not None:
                yield mapped_event
```

### TUIEventMapper Implementation

```python
# New: vibe/core/engine/tui_events.py

class TUIEventMapper:
    """Map native LangGraph events to Vibe TUI events.
    
    This mapper translates events from VibeLangChainEngine's native
    astream_events() output to Vibe TUI event types that
    EventHandler expects.
    
    Supported event types:
    - on_chat_model_stream -> AssistantEvent (token streaming)
    - on_tool_start -> ToolCallEvent (tool execution begins)
    - on_tool_end -> ToolResultEvent (tool execution completes)
    """
    
    def __init__(self, config: VibeConfig) -> None:
        self.config = config
        self.tool_manager = ToolManager(config)

    def map_event(self, event: dict | object) -> BaseEvent | None:
        """Map a native LangGraph event to a Vibe TUI event."""
        # Handle both dict and object types from astream_events
        if not isinstance(event, dict):
            event_data = event.__dict__ if hasattr(event, "__dict__") else {}
        else:
            event_data = event

        event_type = event_data.get("event", "")

        match event_type:
            case "on_chat_model_stream":
                return self._map_chat_model_stream(event_data)
            case "on_tool_start":
                return self._map_tool_start(event_data)
            case "on_tool_end":
                return self._map_tool_end(event_data)
        
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
| **Event Source** | Translated via `EventTranslator` | Native from `astream_events()` + TUIEventMapper |
| **Streaming Modes** | Single mode | Multiple modes (`values`, `updates`, `messages`, `custom`) |
| **Event Types** | Vibe-specific | Native LangGraph events mapped to Vibe events |
| **Token Events** | Via `AssistantEvent` | Via `on_chat_model_stream` -> `AssistantEvent` |
| **Tool Events** | Via `ToolCallEvent`/`ToolResultEvent` | Via `on_tool_start`/`on_tool_end` -> mapped |
| **Code LOC** | ~100 lines in adapter | ~80 lines in TUIEventMapper (with fallback) |

## Implementation Details

### EventTranslator Deprecation

- `EventTranslator` is now deprecated and emits `DeprecationWarning`
- `ApprovalBridge` is also deprecated for the same reason
- Both are kept for backward compatibility with VibeEngine (DeepAgents)

### TUI Integration

- TUI's `EventHandler` remains unchanged (continues to expect `BaseEvent` types)
- VibeLangChainEngine handles mapping internally via `TUIEventMapper`
- Approval flow uses native `Command(resume=...)` pattern with `HumanInTheLoopMiddleware`

## Validation Checklist

- [x] Native events stream correctly
- [x] Token-by-token streaming works (via `on_chat_model_stream`)
- [x] Tool start/end events are captured and mapped
- [x] TUIEventMapper correctly translates events
- [x] EventTranslator and ApprovalBridge emit deprecation warnings
- [x] TUI can consume mapped events (TUIEventHandler unchanged)

## ⚠️ Critical Issues Found (Post-Migration Audit)

1. **Missing Async Middleware Variants** ⚠️
   - All middlewares only implement sync hooks
   - Will crash if LangGraph calls async variants during streaming
   - **Fix Required:** Implement `abefore_model`, `aafter_model`, etc. for all middleware
   - **Implementation:** Delegate to sync versions:
     ```python
     async def abefore_model(self, state: AgentState, runtime: Runtime):
         return self.before_model(state, runtime)

     async def aafter_model(self, state: AgentState, runtime: Runtime):
         return self.after_model(state, runtime)

     async def abefore_agent(self, state: AgentState, runtime: Runtime):
         return self.before_agent(state, runtime)

     async def aafter_agent(self, state: AgentState, runtime: Runtime):
         return self.after_agent(state, runtime)
     ```
   - **Impact:** High - Crashes on async execution path
   - **See Also:** docs/langchain-migration/04-middleware.md for detailed fix

**Note:** This affects all middleware classes (ContextWarningMiddleware, PriceLimitMiddleware, LoggerMiddleware)
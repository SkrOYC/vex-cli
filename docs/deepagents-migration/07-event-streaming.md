# 07 - Event Streaming & Translation

## Overview

Replace Mistral Vibe's custom event system with DeepAgents' LangGraph streaming while maintaining TUI compatibility through event translation.

## Current Event System

### Custom Event Types

```python
# vibe/core/types.py
class BaseEvent(BaseModel): ...

class AssistantEvent(BaseEvent):
    content: str

class ToolCallEvent(BaseEvent):
    tool_call: ToolCall
    tool_config: BaseToolConfig

class ToolResultEvent(BaseEvent):
    tool_call: ToolCall
    result: Any
    success: bool

class CompactStartEvent(BaseEvent):
    current_context_tokens: int
    threshold: int

class CompactEndEvent(BaseEvent):
    old_context_tokens: int
    new_context_tokens: int
    summary_length: int
```

### Event Streaming

```python
# vibe/core/agent.py
async def _conversation_loop(self, user_msg: str) -> AsyncGenerator[BaseEvent]:
    # Manual event generation throughout loop
    yield AssistantEvent(content=chunk.content)
    yield ToolCallEvent(tool_call=tool_call, tool_config=config)
    yield ToolResultEvent(tool_call=tool_call, result=result, success=True)
```

### TUI Event Handling

```python
# vibe/cli/textual_ui/app.py
async def _handle_event(self, event: BaseEvent):
    match event:
        case AssistantEvent():
            self._append_assistant_message(event.content)
        case ToolCallEvent():
            self._show_tool_call(event.tool_call, event.tool_config)
        case ToolResultEvent():
            self._show_tool_result(event.tool_call, event.result, event.success)
```

## DeepAgents Event System

### LangGraph Streaming

```python
# DeepAgents uses LangGraph's built-in streaming
async for event in agent.astream_events(inputs, config, version="v2"):
    match event["event"]:
        case "on_chat_model_start":
            # Model call starting
        case "on_chat_model_stream":
            # Token streaming
            token = event["data"]["chunk"].content
        case "on_chat_model_end":
            # Model call complete
        case "on_tool_start":
            # Tool execution starting
            tool_name = event["name"]
            tool_args = event["data"]["input"]
        case "on_tool_end":
            # Tool execution complete
            tool_result = event["data"]["output"]
```

### Event Structure

```python
{
    "event": "on_tool_start",
    "name": "read_file",
    "run_id": "abc-123",
    "data": {
        "input": {"path": "/file.txt", "offset": 0, "limit": 100}
    },
    "metadata": {...},
    "parent_ids": [...],
    "tags": [...]
}
```

## Migration Strategy

### Phase 1: Event Translator

Create a translation layer between LangGraph events and Vibe events:

```python
# vibe/core/engine/adapters.py

from typing import Any, AsyncGenerator
from langchain_core.messages import AIMessage, ToolMessage

from vibe.core.types import (
    AssistantEvent,
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
    CompactStartEvent,
    CompactEndEvent,
)


class EventTranslator:
    """Translate LangGraph events to Vibe TUI events."""
    
    @staticmethod
    def translate(event: dict[str, Any]) -> BaseEvent | None:
        """Translate a single LangGraph event to Vibe event."""
        event_type = event.get("event")
        event_data = event.get("data", {})
        
        match event_type:
            case "on_chat_model_stream":
                # Streaming token
                chunk = event_data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    return AssistantEvent(content=chunk.content)
            
            case "on_chat_model_end":
                # Full message (non-streaming)
                message = event_data.get("output")
                if isinstance(message, AIMessage) and message.content:
                    return AssistantEvent(content=message.content)
            
            case "on_tool_start":
                # Tool call starting
                tool_name = event.get("name", "")
                tool_args = event_data.get("input", {})
                
                # Create ToolCall for compatibility
                tool_call = ToolCall(
                    id=event.get("run_id", ""),
                    type="function",
                    function=FunctionCall(
                        name=tool_name,
                        arguments=str(tool_args)  # Simplified
                    )
                )
                
                return ToolCallEvent(
                    tool_call=tool_call,
                    tool_config=None  # Will be resolved by TUI
                )
            
            case "on_tool_end":
                # Tool call complete
                tool_name = event.get("name", "")
                tool_result = event_data.get("output", "")
                
                tool_call = ToolCall(
                    id=event.get("run_id", ""),
                    type="function", 
                    function=FunctionCall(name=tool_name, arguments="")
                )
                
                return ToolResultEvent(
                    tool_call=tool_call,
                    result=tool_result,
                    success=True
                )
            
            case "on_chain_start":
                # Middleware or chain starting
                if event.get("name") == "summarize":
                    return CompactStartEvent(
                        current_context_tokens=0,  # Estimate
                        threshold=170000
                    )
            
            case "on_chain_end":
                # Middleware or chain ending
                if event.get("name") == "summarize":
                    return CompactEndEvent(
                        old_context_tokens=0,  # Estimate
                        new_context_tokens=0,  # Estimate
                        summary_length=len(str(event_data.get("output", "")))
                    )
        
        return None


class StreamAdapter:
    """Adapt DeepAgents streaming to Vibe's event interface."""
    
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent
    
    async def run_with_events(
        self, 
        inputs: dict[str, Any], 
        config: dict[str, Any]
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run agent and yield translated events."""
        
        async for event in self.agent.astream_events(inputs, config, version="v2"):
            translated = EventTranslator.translate(event)
            if translated is not None:
                yield translated
```

### Phase 2: TUI Integration

Update TUI to use the new event stream:

```python
# vibe/cli/textual_ui/app.py

class VibeApp(App):
    async def run_conversation(self, user_message: str):
        """Run conversation with event streaming."""
        
        # Get event stream from engine
        async for event in self.engine.run_with_events(user_message):
            await self._handle_event(event)
    
    async def _handle_event(self, event: BaseEvent):
        """Handle translated events (same as before)."""
        match event:
            case AssistantEvent():
                await self._append_assistant_message(event.content)
            case ToolCallEvent():
                await self._show_tool_call(event.tool_call, event.tool_config)
            case ToolResultEvent():
                await self._show_tool_result(event.tool_call, event.result, event.success)
            case CompactStartEvent():
                await self._show_compaction_start(event)
            case CompactEndEvent():
                await self._show_compaction_end(event)
```

### Phase 3: Engine Event Interface

Update VibeEngine to provide event streaming:

```python
# vibe/core/engine/engine.py

class VibeEngine:
    async def run_with_events(
        self, 
        user_message: str
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run conversation and yield events for TUI."""
        if self._agent is None:
            self.initialize()

        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": 1000,
        }

        inputs = {"messages": [("user", user_message)]}
        
        # Use StreamAdapter to translate events
        adapter = StreamAdapter(self._agent)
        async for event in adapter.run_with_events(inputs, config):
            yield event

    async def run(self, user_message: str) -> AsyncGenerator[Any, None]:
        """Legacy interface - yields raw LangGraph events."""
        # For components that need raw events
        async for event in self._agent.astream_events(
            {"messages": [("user", user_message)]},
            {"configurable": {"thread_id": self._thread_id}},
            version="v2",
        ):
            yield event
```

## Event Type Mapping

### Complete Mapping Table

| LangGraph Event | Vibe Event | Notes |
|-----------------|------------|-------|
| `on_chat_model_start` | - | Not needed by TUI |
| `on_chat_model_stream` | `AssistantEvent` | Token streaming |
| `on_chat_model_end` | `AssistantEvent` | Full message (non-streaming) |
| `on_tool_start` | `ToolCallEvent` | Tool execution starting |
| `on_tool_end` | `ToolResultEvent` | Tool execution complete |
| `on_chain_start` (summarize) | `CompactStartEvent` | Context compaction starting |
| `on_chain_end` (summarize) | `CompactEndEvent` | Context compaction complete |
| `on_llm_start` | - | Internal |
| `on_llm_end` | - | Internal |
| `on_prompt_start` | - | Internal |
| `on_prompt_end` | - | Internal |
| `on_retriever_start` | - | Not used |
| `on_retriever_end` | - | Not used |

### Special Cases

#### Tool Call Enrichment

LangGraph events don't include tool config, so enrich in translator:

```python
# In EventTranslator.translate for on_tool_start
tool_config = self._get_tool_config(tool_name)  # From VibeConfig
return ToolCallEvent(tool_call=tool_call, tool_config=tool_config)
```

#### Streaming vs Non-Streaming

Handle both streaming and non-streaming modes:

```python
# Streaming mode
case "on_chat_model_stream":
    return AssistantEvent(content=chunk.content)

# Non-streaming mode  
case "on_chat_model_end":
    return AssistantEvent(content=message.content)
```

## Error Handling

### Event Translation Errors

```python
# vibe/core/engine/adapters.py

class EventTranslator:
    @staticmethod
    def translate(event: dict[str, Any]) -> BaseEvent | None:
        try:
            # Translation logic
            return translated_event
        except Exception as e:
            # Log error but don't crash
            logger.warning(f"Failed to translate event {event.get('event')}: {e}")
            return None
```

### Stream Errors

```python
# vibe/core/engine/engine.py

class VibeEngine:
    async def run_with_events(self, user_message: str):
        try:
            async for event in self._stream_events(user_message):
                yield event
        except Exception as e:
            # Yield error event
            yield ErrorEvent(message=f"Engine error: {e}")
```

## Performance Considerations

### Event Filtering

Only translate events that TUI needs:

```python
# Filter out internal events
IGNORED_EVENTS = {
    "on_llm_start", "on_llm_end", 
    "on_prompt_start", "on_prompt_end",
    "on_retriever_start", "on_retriever_end"
}

if event_type in IGNORED_EVENTS:
    return None
```

### Buffering for Efficiency

Buffer rapid events to prevent UI flooding:

```python
# Group rapid token events
class EventBuffer:
    def __init__(self, max_delay: float = 0.1):
        self.buffer: list[BaseEvent] = []
        self.max_delay = max_delay
        self.last_yield = time.time()
    
    async def add_event(self, event: BaseEvent):
        self.buffer.append(event)
        
        if time.time() - self.last_yield > self.max_delay:
            await self.flush()
    
    async def flush(self):
        if self.buffer:
            # Yield combined event or individual events
            for event in self.buffer:
                yield event
            self.buffer.clear()
            self.last_yield = time.time()
```

## Files to Modify

### Core Engine
- `vibe/core/engine/adapters.py` - Add EventTranslator and StreamAdapter
- `vibe/core/engine/engine.py` - Add event streaming interface

### TUI Integration  
- `vibe/cli/textual_ui/app.py` - Update to use new event stream
- `vibe/cli/textual_ui/widgets/messages.py` - Handle new event types

### Remove Old Event System
- `vibe/core/agent.py` - Remove manual event generation
- `vibe/core/types.py` - Keep event types, remove custom streaming logic

## Validation Checklist

- [ ] LangGraph events translate to Vibe events correctly
- [ ] Streaming tokens display in real-time
- [ ] Tool calls show with proper formatting
- [ ] Tool results display correctly
- [ ] Compaction events trigger UI updates
- [ ] Error events handled gracefully
- [ ] Performance acceptable (no UI lag)
- [ ] Both streaming and non-streaming modes work
- [ ] Event buffering prevents UI flooding
- [ ] Backward compatibility maintained

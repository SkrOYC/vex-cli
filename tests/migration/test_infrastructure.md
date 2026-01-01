# LangChain Migration Test Infrastructure Guide

This document explains how to use the new test infrastructure introduced during the
migration from legacy Agent class to VibeLangChainEngine.

## Overview

The test infrastructure has been updated to use `FakeVibeLangChainEngine` instead of
the legacy `Agent` class. This provides deterministic, isolated test execution without
requiring real LLM calls or LangGraph state management.

## Key Components

### 1. FakeVibeLangChainEngine

Located in `tests/stubs/fake_backend.py`, this mock implements the `EngineInterface`
protocol and provides:

- **Event streaming**: Yields synthetic `BaseEvent` instances (AssistantEvent, ToolCallEvent, ToolResultEvent)
- **Stats management**: Maintains `VibeEngineStats` that matches production behavior
- **State tracking**: Supports `compact()`, `clear_history()`, and session management
- **HITL support**: Integrates with `FakeInterruptedAgent` for approval testing

### 2. Test Fixtures

Located in `tests/conftest.py`, these fixtures provide pre-configured engines:

- `fake_engine_basic_response()`: Engine that yields a simple AssistantEvent
- `fake_engine_with_tool_call()`: Engine with a full tool execution flow
- `fake_engine_with_compact()`: Engine for testing compact functionality
- `fake_engine_with_events()`: Generic fixture for custom event sequences

### 3. Mock Utilities

Located in `tests/mock/utils.py`, these utilities help create test data:

- `create_mock_hitl_request()`: Create HITLRequest for testing approval flows
- `create_mock_hitl_response()`: Create HITLResponse for testing resume decisions
- `create_mock_multi_tool_response()`: Create multi-tool approval responses
- `mock_llm_chunk()`: Create LLMChunk for synthetic responses

## Usage Patterns

### Pattern 1: Basic Conversation Flow

```python
@pytest.mark.asyncio
async def test_basic_conversation(fake_engine_basic_response):
    events = []
    async for event in fake_engine_basic_response.run("Hello"):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], AssistantEvent)
    assert events[0].content == "This is a test response"
```

### Pattern 2: Tool Execution Flow

```python
@pytest.mark.asyncio
async def test_tool_execution(fake_engine_with_tool_call):
    events = []
    async for event in fake_engine_with_tool_call.run("Execute"):
        events.append(event)

    assert isinstance(events[0], AssistantEvent)
    assert isinstance(events[1], ToolCallEvent)
    assert isinstance(events[2], ToolResultEvent)
```

### Pattern 3: Stats Verification

```python
@pytest.mark.asyncio
async def test_stats_accumulation():
    engine = FakeVibeLangChainEngine(
        config=make_config(),
        events_to_yield=[
            AssistantEvent(content="Response 1"),
            AssistantEvent(content="Response 2"),
        ],
    )
    engine.initialize()

    # Run multiple turns
    async for _ in engine.run("Turn 1"):
        pass
    async for _ in engine.run("Turn 2"):
        pass

    assert engine.stats.steps == 2
```

### Pattern 4: HITL Approval Testing

```python
@pytest.mark.asyncio
async def test_tool_approval():
    engine = FakeVibeLangChainEngine(
        config=make_config(),
        events_to_yield=[
            ToolCallEvent(tool_name="bash", args={"command": "ls"}, ...),
        ],
    )

    approvals = [True]
    feedbacks = [None]

    await engine.handle_multi_tool_approval(approvals, feedbacks)

    assert engine.stats.tool_calls_agreed == 1
```

### Pattern 5: Compact Testing

```python
@pytest.mark.asyncio
async def test_compact():
    engine = FakeVibeLangChainEngine(
        config=make_config(),
        events_to_yield=[
            AssistantEvent(content=f"Message {i}") for i in range(5)
        ],
    )
    engine.initialize()

    # Populate engine state
    async for _ in engine.run("Initial"):
        pass

    initial_count = len(engine._events)
    result = engine.compact()

    assert "Compacted" in result
    assert len(engine._events) < initial_count
```

## Advantages Over Legacy Approach

1. **No Network Dependency**: Tests run completely offline
2. **Deterministic Execution**: Same inputs always produce same outputs
3. **Fast Execution**: No LLM API latency
4. **Complete Control**: Can test edge cases and error conditions
5. **LangChain Native**: Tests use actual LangChain event types and structures

## Migration Checklist

When migrating legacy tests to new infrastructure:

- [ ] Replace `Agent(config, backend=backend)` with `FakeVibeLangChainEngine(config=config)`
- [ ] Update `agent.act()` calls to `agent.run()`
- [ ] Remove `FakeBackend` dependency if only checking events
- [ ] Update assertions to use `engine.stats` instead of `agent.stats`
- [ ] Replace `agent.messages` with event sequence inspection
- [ ] Update `AgentStats` references to `VibeEngineStats`
- [ ] Remove or update `reload_with_initial_messages()` tests
- [ ] Update `compact()` tests to use `engine.compact()` directly

## Common Pitfalls

### Pitfall 1: Forgetting to Initialize

```python
# WRONG
engine = FakeVibeLangChainEngine(config=config)
async for event in engine.run("Hello"):  # Will fail!
    pass

# CORRECT
engine = FakeVibeLangChainEngine(config=config)
engine.initialize()
async for event in engine.run("Hello"):
    pass
```

### Pitfall 2: Not Handling AsyncGenerator

```python
# WRONG - Tries to collect all events at once
events = list(engine.run("Hello"))  # Doesn't work with async generators!

# CORRECT - Iterate through the async generator
events = []
async for event in engine.run("Hello"):
    events.append(event)
```

### Pitfall 3: Expecting Agent-Specific Methods

```python
# WRONG - Agent class had these methods
engine.auto_approve = True
engine.messages.append(...)

# CORRECT - Use VibeLangChainEngine's interface
await engine.run(message)
assert engine.stats.steps == expected
```

## Testing with LangChain Events

The `FakeVibeLangChainEngine` streams synthetic events that match the structure of
real LangGraph events. Use the helper functions to create proper event structures:

```python
from tests.mock.utils import create_mock_llm_chunk

# Create AssistantEvent
from vibe.core.types import AssistantEvent
event = AssistantEvent(content="Response text")

# Create ToolCallEvent with proper args
from vibe.core.types import ToolCallEvent
event = ToolCallEvent(
    tool_name="bash",
    args={"command": "echo test"},
    tool_call_id="call-123",
    tool_class=BashTool,
)

# Create ToolResultEvent
from vibe.core.types import ToolResultEvent
event = ToolResultEvent(
    tool_name="bash",
    tool_call_id="call-123",
    result={"output": "test output"},
    tool_class=BashTool,
)
```

## Integration with Existing Tests

The new mock infrastructure is designed to work with existing test patterns:

- **Snapshot tests**: `tests/snapshots/` can use `FakeVibeLangChainEngine` directly
- **Unit tests**: Use fixtures from `conftest.py` for consistent configuration
- **Integration tests**: Test TUI integration with fake engine providing deterministic events
- **Performance tests**: Remove LLM latency from measurements by using synthetic events

## Future Enhancements

Potential improvements to test infrastructure:

1. **Event Recording**: Add mode to record all events for assertion
2. **State Diffing**: Compare LangGraph state before/after operations
3. **Middleware Testing**: Create specific fixtures for testing custom middleware
4. **Error Simulation**: Add fixtures for testing error conditions
5. **Concurrency Testing**: Test multi-session and concurrent request handling

## References

- LangChain Agents Documentation: https://python.langchain.com/docs/modules/agents/
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- Test Utilities: `tests/mock/utils.py`
- Mock Implementations: `tests/stubs/fake_backend.py`

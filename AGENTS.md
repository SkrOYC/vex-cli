# Development Commands

**All commands must run under `nix develop .#fhs --command`**

## Package Management
```bash
nix develop .#fhs --command uv sync                    # Install dependencies
nix develop .#fhs --command uv sync --all-extras       # Install all extras
nix develop .#fhs --command uv add <package>           # Add dependency
nix develop .#fhs --command uv remove <package>        # Remove dependency
```

## Running Commands
```bash
# Use python directly (recommended) - avoids .venv conflicts
nix develop .#fhs --command python -m pytest                          # All tests
nix develop .#fhs --command python -m pytest --ignore tests/snapshots  # Exclude snapshots
nix develop .#fhs --command python -m pytest tests/snapshots          # Snapshots only
nix develop .#fhs --command python -m pytest tests/performance/test_benchmarks.py  # Benchmarks
nix develop .#fhs --command python -m pytest <file>::<test>           # Single test
nix develop .#fhs --command python -m pytest -v -x -n 0               # Verbose, stop on fail, no parallel

# Alternative: Use uv run (may have .venv conflicts in some environments)
nix develop .#fhs --command uv run pytest                          # All tests
nix develop .#fhs --command uv run python -m vibe --prompt "test"  # Run vibe CLI
```

## Linting & Formatting
```bash
nix develop .#fhs --command python -m ruff check --fix --unsafe-fixes  # Lint + fix
nix develop .#fhs --command python -m ruff format                      # Format code
nix develop .#fhs --command python -m pyright                          # Type check
nix develop .#fhs --command python -m pre_commit run --all-files       # Run hooks
nix develop .#fhs --command python -m typos --write-changes            # Spell check
```

## CLI Verification
```bash
# Use python directly to run vibe CLI
nix develop .#fhs --command python -m vibe --help
nix develop .#fhs --command python -m vibe --prompt "Say hello" --output text
```

## Nix Operations
```bash
nix develop .#fhs                              # Enter interactive env
nix flake check                                # Check flake
```

# Code Style Guidelines

- **Python 3.12+**: Start files with `from __future__ import annotations`
- **Type hints**: `list[str]`, `dict[str, int]`, `int | None`; annotate all public APIs; no inline `# type: ignore`
- **Ruff**: 88 char lines; absolute imports only; known first-party: `vibe`
- **Naming**: Classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`, private `_underscore`, enums `UPPERCASE`
- **Modern syntax**: `match-case`, walrus `:=`, f-strings, comprehensions
- **Paths**: Always use `pathlib.Path`, not `os.path`
- **Enums**: StrEnum for strings, IntEnum/IntFlag for ints; use `auto()`, UPPERCASE members; use `@property` for computed
- **Pydantic v2**: `model_validate()`, `field_validator`, `model_dump()`, `from_attributes=True`
- **Pydantic unions**: Sibling classes + mixin; `Annotated[Union[...], Field(discriminator='transport')]`
- **Errors**: Custom from `RuntimeError`/`ValueError`; include attrs; only document explicitly raised exceptions
- **Structure**: Early returns; max 50 statements, 15 branches, 15 locals, 9 args, 6 returns
- **Testing**: Fixtures in `tests/conftest.py`; mock externals; descriptive names
- **Docs**: Google-style docstrings for public APIs

**Workflow**: Make changes → lint → type check → `pytest -x`

# LangChain Migration Testing Guidelines

## Test Infrastructure Overview

As of LangChain 1.2.0 migration, the test infrastructure uses:
- `FakeVibeLangChainEngine`: Deterministic mock of `VibeLangChainEngine` for offline testing
- `EngineInterface` protocol: Common interface that both real and fake engines implement
- `VibeEngineStats`: Statistics tracking used by both production and test engines

## Using FakeVibeLangChainEngine

The `FakeVibeLangChainEngine` class (in `tests/stubs/fake_backend.py`) provides:
- Deterministic event streaming (no random LLM calls)
- Full `EngineInterface` implementation for TUI compatibility
- Configurable synthetic event sequences for various test scenarios
- Stats tracking that matches production behavior

**Example:**
```python
from tests.stubs.fake_backend import FakeVibeLangChainEngine

engine = FakeVibeLangChainEngine(
    config=test_config,
    events_to_yield=[
        AssistantEvent(content="Hello"),
        ToolCallEvent(tool_name="bash", args={"command": "echo test"}),
        ToolResultEvent(tool_name="bash", result={"output": "test"}),
    ],
)
engine.initialize()

async for event in engine.run("Test message"):
    process_event(event)
```

## Test Fixtures

Use fixtures from `tests/conftest.py` for consistent test configuration:

- `fake_engine_basic_response()`: Simple assistant response
- `fake_engine_with_tool_call()`: Complete tool execution flow
- `fake_engine_with_compact()`: For testing compact functionality
- `fake_engine_with_events()`: Generic fixture for custom event sequences

## Testing Patterns

### Pattern 1: Basic Conversation
```python
async def test_conversation():
    engine = fake_engine_basic_response()
    events = []
    async for event in engine.run("Hello"):
        events.append(event)
    assert isinstance(events[0], AssistantEvent)
```

### Pattern 2: Tool Execution
```python
async def test_tool_execution():
    engine = fake_engine_with_tool_call()
    events = []
    async for event in engine.run("Execute command"):
        events.append(event)
    assert any(isinstance(e, ToolCallEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) for e in events)
```

### Pattern 3: Stats Verification
```python
async def test_stats():
    engine = fake_engine_basic_response()
    await engine.run("Test")
    assert engine.stats.steps == 1
    assert engine.stats.session_cost >= 0
```

### Pattern 4: Compact Functionality
```python
async def test_compact():
    engine = fake_engine_with_compact()
    await engine.run("Test")
    result = engine.compact()
    assert "Compacted" in result
    assert len(engine._events) < 5  # Reduced from 5 to ~2
```

## Migration Checklist for Tests

When updating tests from legacy `Agent` to `VibeLangChainEngine`:

- [ ] Replace `Agent(config, backend=backend)` with `FakeVibeLangChainEngine(config=config)`
- [ ] Update `agent.act()` calls to `agent.run()`
- [ ] Update `agent.messages` to use event streaming
- [ ] Replace `AgentStats` with `VibeEngineStats`
- [ ] Update stats assertions to use `engine.stats` property
- [ ] Remove `FakeBackend` usage if not needed
- [ ] Update `reload_with_initial_messages()` to use LangGraph state
- [ ] Verify all tests work offline (no API keys required)

## Real-Time Stats Updates

The `VibeLangChainEngine` now implements real-time stats updates during event streaming. This ensures the UI displays accurate statistics during agent execution, fixing the issue where stats were only updated on-demand.

### How It Works

1. **Incremental Updates**: Stats are updated incrementally from LangGraph events during the `run()` method's event streaming loop
2. **Event-Driven**: `on_chat_model_end` events provide token usage via `usage_metadata`, while `on_tool_end` events track step completion
3. **Lightweight Property Access**: The `stats` property now simply returns the current stats state without expensive recomputation

### Implementation Details

**Production Code** (`vibe/core/engine/langchain_engine.py`):
```python
async def run(self, user_message: str) -> AsyncGenerator[Any, None]:
    async for event in self._agent.astream_events(...):
        # Update stats incrementally from event data
        self._update_stats_from_event(event)
        
        mapped_event = mapper.map_event(event)
        if mapped_event is not None:
            yield mapped_event

def _update_stats_from_event(self, event: dict[str, Any]) -> None:
    event_type = event.get("event", "")
    
    if event_type == "on_chat_model_end":
        # Extract usage_metadata from AIMessage output
        output = event.get("data", {}).get("output", {})
        if hasattr(output, "usage_metadata") and output.usage_metadata:
            usage = output.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
            # Update session and last turn token counts
            self._stats.session_prompt_tokens += input_tokens
            self._stats.session_completion_tokens += output_tokens
            self._stats.last_turn_prompt_tokens = input_tokens
            self._stats.last_turn_completion_tokens = output_tokens
            
            # Update context tokens incrementally
            self._stats.context_tokens += input_tokens + output_tokens
    
    elif event_type == "on_tool_end":
        self._stats.steps += 1
```

**Test Infrastructure** (`tests/stubs/fake_backend.py`):
- `FakeVibeLangChainEngine` mirrors this behavior with `_update_stats_from_vibe_event()` method
- Updates stats before yielding events to ensure accurate stats during streaming

### Testing Real-Time Updates

Use the test pattern to verify real-time stats updates:

```python
async def test_stats_update_during_run(self) -> None:
    engine = FakeVibeLangChainEngine(
        config=make_config(),
        events_to_yield=[
            AssistantEvent(content="Response 1"),
            AssistantEvent(content="Response 2"),
        ],
    )
    engine.initialize()

    # Track stats during streaming
    stats_during_run = []
    async for event in engine.run("Test"):
        stats_during_run.append(engine.stats.steps)

    # Stats should have been updated during streaming
    assert stats_during_run[0] == 1  # First event processed
    assert stats_during_run[1] == 2  # Second event processed
```

## Common Pitfalls to Avoid

### Pitfall 1: Not Initializing Fake Engine
```python
# WRONG - Will fail at runtime
engine = FakeVibeLangChainEngine(config=test_config)
async for event in engine.run("Hello"):  # Error!
    pass

# CORRECT - Always initialize before using
engine = FakeVibeLangChainEngine(config=test_config)
engine.initialize()
async for event in engine.run("Hello"):
    pass
```

### Pitfall 2: Collecting Async Generator Incorrectly
```python
# WRONG - Tries to collect all events at once
events = list(engine.run("Hello"))  # Doesn't work!

# CORRECT - Iterate through the async generator
events = []
async for event in engine.run("Hello"):
    events.append(event)
```

### Pitfall 3: Using Legacy Agent Methods
```python
# WRONG - Agent class had these methods
agent.auto_approve = True
agent.messages.append(...)

# CORRECT - Use VibeLangChainEngine's interface
async for event in engine.run(message):
    process_event(event)
assert engine.stats.steps == expected
```

## Testing with Mock Models

For integration tests requiring LLM behavior (not just event streaming):

```python
from tests.mock.utils import get_mocking_env

# Set up mock responses
env = get_mocking_env([
    mock_llm_chunk(content="Response 1"),
    mock_llm_chunk(content="Response 2"),
])
with env:
    # Test with mock model
    pass
```

## Documentation Reference

- Test Infrastructure Guide: `tests/migration/test_infrastructure.md`
- Mock Utilities: `tests/mock/utils.py`
- Mock Implementations: `tests/stubs/fake_backend.py`
- Test Fixtures: `tests/conftest.py`
- Type Definitions: `vibe/core/types.py`

# Implementation Notes

## FHS Environment Requirement
All tests and commands require `libstdc++.so.6` which is only available in FHS environment.

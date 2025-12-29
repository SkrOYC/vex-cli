# 10 - Testing Strategy

## Overview

Comprehensive testing strategy to ensure feature parity, performance, and reliability during and after migration.

## Test Structure

```
tests/
├── unit/
│   ├── test_langchain_engine.py      # NEW
│   ├── test_langchain_middleware.py  # NEW
│   ├── test_tools.py                 # UPDATE
│   └── test_config.py                # Minor updates
├── integration/
│   ├── test_langchain_integration.py # NEW
│   ├── test_approval_flow.py         # UPDATE (native HITL)
│   └── test_tool_execution.py        # Minor updates
└── migration/
    ├── test_deepagents_parity.py     # NEW
    └── test_langchain_features.py    # NEW
```

## Key Test Cases

### Unit Tests

```python
# tests/unit/test_langchain_engine.py

def test_engine_initialization():
    """Test VibeLangChainEngine initializes correctly."""
    config = VibeConfig(...)
    engine = VibeLangChainEngine(config)
    assert engine._agent is None
    assert engine._thread_id is not None

def test_middleware_stack():
    """Test middleware stack building."""
    engine = VibeLangChainEngine(config)
    middleware = engine._build_middleware_stack()
    assert len(middleware) == 3  # Context, Price, HITL

def test_context_warning_middleware():
    """Test context warning middleware."""
    middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=100000)
    state = {"messages": [...], "context_tokens": 60000}
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert "warning" in result

def test_price_limit_middleware():
    """Test price limit middleware."""
    middleware = PriceLimitMiddleware(max_price=1.00, pricing={"model": (0.000001, 0.000003)})
    state = {"messages": [AIMessage(content="test", usage_metadata={...})]}
    result = middleware.after_model(state, runtime)
    assert result is None
```

### Integration Tests

```python
# tests/integration/test_langchain_integration.py

@pytest.mark.asyncio
async def test_conversation_flow():
    """Test complete conversation flow with LangChain engine."""
    config = VibeConfig(workdir=tmp_path)
    engine = VibeLangChainEngine(config)
    
    # Initialize
    engine.initialize()
    
    # Run conversation
    events = []
    async for event in engine.run("List files in current directory"):
        events.append(event)
    
    # Verify events
    assert len(events) > 0
    assert any(e["event"] == "on_tool_start" for e in events)
    assert any(e["event"] == "on_tool_end" for e in events)

@pytest.mark.asyncio
async def test_approval_workflow():
    """Test approval workflow with native HITL."""
    config = VibeConfig(workdir=tmp_path)
    approval_future = asyncio.Future()
    
    engine = VibeLangChainEngine(
        config=config,
        approval_callback=lambda req: approval_future,
    )
    
    # Run with approval-requiring tool
    task = asyncio.create_task(collect_events(engine, "run dangerous command"))
    
    # Wait for interrupt
    await asyncio.wait_for(approval_future, timeout=5.0)
    
    # Approve
    await engine.handle_approval(True, "request-id", "Approved")
    
    # Verify completion
    events = await task
    assert any(e["event"] == "interrupt" for e in events)
```

### Migration Parity Tests

```python
# tests/migration/test_deepagents_parity.py

@pytest.mark.asyncio
async def test_same_outputs():
    """Test that LangChain engine produces same outputs as DeepAgents."""
    # Run same conversation with both engines
    events_deepagents = await run_with_engine("deepagents", "Explain this code")
    events_langchain = await run_with_engine("langchain", "Explain this code")
    
    # Compare key outputs (token counts, tool calls, etc.)
    assert extract_tokens(events_deepagents) == extract_tokens(events_langchain)
    assert extract_tool_calls(events_deepagents) == extract_tool_calls(events_langchain)
```

## Test Coverage Goals

| Category | Target Coverage | Notes |
|----------|----------------|-------|
| Unit Tests | 90% | All middleware, engine methods |
| Integration Tests | 85% | Full workflows, approval flow |
| Migration Tests | 100% | Parity, edge cases |
| Performance Tests | Baseline | Ensure no regression |

## Migration Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Feature parity with DeepAgents verified
- [ ] Performance baseline established
- [ ] No regressions in TUI
- [ ] All skipped tests are addressed
- [ ] Documentation updated

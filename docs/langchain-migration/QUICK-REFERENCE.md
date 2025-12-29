# LangChain Migration - Quick Reference

## Overview

This document provides a quick reference for the LangChain 1.2.0 migration, listing key changes and decisions.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Agent Creation** | `create_agent()` with middleware API | Simplicity over maximum flexibility |
| **Checkpoint Storage** | InMemory (user's custom SQLite later) | CLI context, no external dependencies |
| **Migration Scope** | Full DeepAgents removal | Clean architecture, no dual maintenance |
| **Backward Compatibility** | Complete cutover | No need for fallback complexity |

## Files to Remove

```python
# Dependencies to remove from pyproject.toml
deepagents = ">=0.3.0"  # Full removal

# Code to remove
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware import ...

# Adapter layers to remove
vibe/core/engine/adapters.py  # EventTranslator + ApprovalBridge
```

## Files to Create

```
vibe/core/engine/
├── langchain_engine.py       # NEW: VibeLangChainEngine
├── langchain_middleware.py   # NEW: Custom middleware
├── state.py                  # NEW: State schema
└── checkpoint.py             # NEW: Checkpoint management
```

## Phase Order

1. **Dependencies** - Update pyproject.toml
2. **Engine** - Create VibeLangChainEngine
3. **Middleware** - Build custom middleware stack
4. **Tools** - Update VibeToolAdapter
5. **State** - Implement state schema
6. **Approval** - Native HITL integration
7. **Events** - Native LangGraph streaming
8. **TUI** - Connect to new engine
9. **Testing** - Comprehensive validation

## Key Imports

### Before (DeepAgents)

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
```

### After (LangChain 1.2.0)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
```

## Quick Validation

Run these commands after each phase:

```bash
# Check for DeepAgents imports
grep -r "from deepagents" vibe/

# Check for adapter usage
grep -r "EventTranslator\|ApprovalBridge" vibe/

# Run tests
uv run pytest tests/unit/test_langchain_engine.py -v
uv run pytest tests/integration/test_langchain_integration.py -v
```

## Timeline

**Total Timeline: 9-11 weeks**

| Phase | Duration | Total |
|-------|----------|-------|
| **1. Dependencies** | 1 week | Week 1 |
| **2. Engine** | 2-3 weeks | Weeks 2-4 |
| **3. Middleware** | 2 weeks | Weeks 3-5 |
| **4. Tools** | 1 week | Week 5 |
| **5. State** | 1-2 weeks | Weeks 6-7 |
| **6. Approval** | 1 week | Week 7 |
| **7. Events** | 1 week | Week 8 |
| **8. TUI** | 1 week | Week 8 |
| **9. Testing** | 2 weeks | Weeks 9-10 |

## Success Criteria

- [ ] No DeepAgents imports remain
- [ ] All adapter layers removed
- [ ] Token tracking is native (no estimation)
- [ ] Test coverage > 85%
- [ ] Performance baseline met
- [ ] TUI works unchanged

## Questions

1. **Python Version**: 3.12+ (confirmed)
2. **LangChain Version**: 1.2.0 (confirmed)
3. **Documentation**: Full docs in `docs/langchain-migration/` (completed)
4. **Testing**: Tests separate from documentation (confirmed)
5. **User Testing**: Begin after Phase 2 (confirmed)

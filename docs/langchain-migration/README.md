# LangChain Migration Plan

## Executive Summary

This document outlines a comprehensive plan to migrate Mistral Vibe from DeepAgents' opinionated architecture to direct **LangChain 1.2.0** integration using `create_agent()` and the new middleware API, gaining full control over middleware, configuration, and agent behavior while preserving the rich Textual TUI experience.

## Vision

**Before**: DeepAgents `create_deep_agent()` + adapter layers (`EventTranslator`, `ApprovalBridge`) + workarounds for opinionated defaults

**After**: LangChain 1.2.0 `create_agent()` + native middleware + full configuration control

**Result**: Complete architectural flexibility + simplified codebase + native LangChain features without adapter overhead

## Key Principles

1. **LangChain 1.2.0 as Engine**: Use `create_agent()` with middleware API as the core execution engine
2. **Preserve TUI**: Keep Mistral Vibe's Textual UI as the presentation layer - no changes to user experience
3. **Remove Adapters**: Eliminate `EventTranslator` and `ApprovalBridge` by using native LangGraph patterns
4. **Full Customization**: Build middleware stack from scratch without DeepAgents' forced defaults
5. **Maintain Compatibility**: All existing configuration, tools, and features continue to work

## Migration Overview

### What Changes

| Component | Before | After |
|-----------|--------|-------|
| **Agent Engine** | `create_deep_agent()` + adapter layers | `create_agent()` + native middleware |
| **Middleware** | DeepAgents-bundled with workarounds | Custom LangChain `AgentMiddleware` stack |
| **Approval** | `ApprovalBridge` adapter | Native `HumanInTheLoopMiddleware` |
| **Events** | `EventTranslator` adapter | Native LangGraph streaming events |
| **State** | Manual token estimation | Native `AgentState` with usage_metadata |
| **Checkpoints** | Basic InMemorySaver | Full checkpointing support |

### What Stays the Same

- **TUI Experience**: Textual interface, approval dialogs, keyboard shortcuts
- **Configuration**: TOML files, project detection, environment variables
- **User Workflows**: Tool usage, conversation flow, project context
- **API Surface**: Configuration loading, CLI commands, plugin system
- **Tool System**: Existing tools continue to work with thin adapters

### New Capabilities Gained

- **Custom Middleware Stack**: No forced defaults, compose exactly what's needed
- **Native State Management**: Proper `TypedDict` schemas with reducers
- **Flexible Checkpointing**: Ready for user's custom SQLite implementation
- **Advanced Streaming**: Access all LangGraph streaming modes (`values`, `updates`, `messages`, `custom`)
- **Configuration Layering**: Full `RunnableConfig` control over execution
- **Direct LangGraph Access**: Customize graph structure when needed

## Architecture Comparison

### Current Architecture (DeepAgents)

```
┌───────────────────────────────────────────────────────────────┐
│                  Mistral Vibe TUI (Preserved)                 │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                    Adapter Layer (Overhead)                    │
│       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│       │   Event     │  │  Approval   │  │   Config    │        │
│       │ Translator  │  │   Bridge    │  │   Loader    │        │
│       └─────────────┘  └─────────────┘  └─────────────┘        │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DeepAgents Engine (Opinionated)                 │
│       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│       │create_deep  │  │Filesystem   │  │  TodoList   │         │
│       │  _agent()   │  │  Backend    │  │ Middleware  │         │
│       └─────────────┘  └─────────────┘  └─────────────┘         │
│       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│       │ SubAgent    │  │Summarization│  │    HITL     │         │
│       │ Middleware  │  │ Middleware  │  │ Middleware  │         │
│       └─────────────┘  └─────────────┘  └─────────────┘         │
│                    ^ FORCED DEFAULTS ^                          │
└─────────────────────────────────────────────────────────────────┘
```

### Target Architecture (LangChain 1.2.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mistral Vibe TUI (Preserved)                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Thin Adapter Layer (Minimal)                 │
│               ┌─────────────┐  ┌─────────────┐                  │
│               │   Config    │  │   Tool      │                  │
│               │   Loader    │  │  Adapter    │                  │
│               └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LangChain 1.2.0 Engine (Flexible)            │
│        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│        │create_agent │  │ Custom      │  │  HumanInThe │        │
│        │  ()         │  │ Middleware  │  │   Loop      │        │
│        └─────────────┘  └─────────────┘  └─────────────┘        │
│        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│        │  StateGraph │  │ Checkpoint  │  │  Streaming  │        │
│        │  (optional) │  │   Saver     │  │   Events    │        │
│        └─────────────┘  └─────────────┘  └─────────────┘        │
│                         ^ YOUR CHOICE ^                         │
└─────────────────────────────────────────────────────────────────┘
```

## Code Impact Analysis

### Current Pain Points

| Component | Current LOC | Issue |
|-----------|-------------|-------|
| `vibe/core/engine/engine.py` | ~405 lines | Adapter overhead, token estimation, rough compaction |
| `vibe/core/engine/adapters.py` | ~256 lines | EventTranslator and ApprovalBridge |
| `vibe/core/engine/middleware.py` | ~204 lines | Workarounds for DeepAgents defaults |
| Token estimation | Multiple locations | Character-based approximation (`// 4`) |
| Summarization | Config only | No implementation, falls back to truncation |

### Target State

| Component | Target LOC | Improvement |
|-----------|------------|-------------|
| `vibe/core/engine/langchain_engine.py` | ~200 lines | 50% reduction |
| `vibe/core/engine/adapters.py` | Removed | 100% removal |
| `vibe/core/engine/langchain_middleware.py` | ~150 lines | Custom stack, no workarounds |
| Token tracking | Native | Actual usage_metadata from LLM |

### Total Estimated Reduction: ~600 lines (40% reduction in engine code)

## Migration Phases

### Phase 1: Foundation & Dependencies (1 week)

**Goals:**
- Add direct LangChain/LangGraph dependencies
- Create abstraction layer for agent creation
- Establish feature flag for gradual rollout

**Deliverables:**
- Updated `pyproject.toml` with LangChain 1.2.0 dependencies
- New `vibe/core/engine/langchain_engine.py` module
- Feature flag `USE_LANGCHAIN` for parallel testing

### Phase 2: Agent Engine Implementation (2-3 weeks)

**Goals:**
- Replace `create_deep_agent()` with `create_agent()`
- Implement custom middleware stack
- Remove `EventTranslator` adapter

**Deliverables:**
- `VibeLangChainEngine` class
- Custom middleware implementations
- Native event streaming integration

### Phase 3: Middleware Migration (2 weeks)

**Goals:**
- Migrate custom middleware to LangChain patterns
- Implement new middleware capabilities
- Remove DeepAgents middleware workarounds

**Deliverables:**
- `vibe/core/engine/langchain_middleware.py`
- Complete middleware stack implementation
- Validation test suite

### Phase 4: Tool Integration (1 week)

**Goals:**
- Migrate `VibeToolAdapter` for native LangChain
- Remove DeepAgents tool dependencies
- Support all existing tool types

**Deliverables:**
- Updated tool adapter
- MCP tool integration
- Custom tool loading

### Phase 5: State Management & Checkpointing (1-2 weeks)

**Goals:**
- Implement proper state schema with TypedDict
- Add durable checkpointing options
- Enable time-travel and state recovery

**Deliverables:**
- Custom `AgentState` schema
- Checkpoint configuration options
- State inspection utilities

### Phase 6: Approval System Migration (1 week)

**Goals:**
- Replace `ApprovalBridge` with native `HumanInTheLoopMiddleware`
- Simplify approval flow
- Remove interrupt translation layer

**Deliverables:**
- Native approval integration
- Simplified event handling
- Updated TUI integration

### Phase 7: Event Streaming (1 week)

**Goals:**
- Use native LangGraph streaming
- Remove `EventTranslator`
- Support all streaming modes

**Deliverables:**
- Native event streaming
- Multiple streaming mode support
- Simplified event handling

### Phase 8: TUI Integration (1 week)

**Goals:**
- Update TUI to use new engine
- Remove adapter dependencies
- Preserve all UX behaviors

**Deliverables:**
- Updated TUI integration
- Preserved user experience
- Simplified event handling

### Phase 9: Testing & Validation (2 weeks)

**Goals:**
- Comprehensive test suite
- Feature parity validation
- Performance benchmarking
- User acceptance testing

**Deliverables:**
- Unit tests for all components
- Integration tests for workflows
- Migration validation checklist
- User testing and feedback

## Success Metrics

### Code Quality

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Engine LOC | ~405 | ~200 | 50% reduction |
| Adapter LOC | ~256 | 0 | 100% removal |
| Token estimation | Manual | Native | 100% accuracy |
| Event Handling | Translated | Native | Direct access |

### Feature Parity

| Feature | Before | After | Status |
|---------|--------|-------|--------|
| Context warnings | ✅ | ✅ | Native |
| Price limits | ✅ | ✅ | Native |
| Approval flow | ✅ (Adapter) | ✅ (Native) | Simplified |
| Tool execution | ✅ | ✅ | Native |
| Streaming events | ✅ (Translated) | ✅ (Native) | Direct |
| Checkpointing | Basic | Advanced | Ready |

### Compatibility

| Aspect | Status |
|--------|--------|
| Configuration files | 100% compatible |
| TUI experience | 100% preserved |
| Tool system | 100% compatible |
| CLI commands | No changes |

## Risk Mitigation

### Gradual Rollout

- Feature flag for each migration phase
- Parallel old/new implementation testing
- Ability to rollback any phase
- A/B testing with users

### Comprehensive Testing

- Unit tests for all components
- Integration tests for workflows
- Performance regression tests
- Manual validation checklists

### Backward Compatibility

- All existing config files work
- CLI commands preserved
- Plugin system maintained
- API surface unchanged

## Document Index

### Core Migration

- **[00-overview.md](00-overview.md)** - Architecture overview and principles
- **[01-dependencies.md](01-dependencies.md)** - Package dependencies and compatibility
- **[02-agent-engine.md](02-agent-engine.md)** - Core agent replacement strategy
- **[03-tools.md](03-tools.md)** - Tool migration approach
- **[04-middleware.md](04-middleware.md)** - Middleware implementation plan
- **[05-state-management.md](05-state-management.md)** - State schema and checkpointing
- **[06-approval-system.md](06-approval-system.md)** - Native HITL integration
- **[07-event-streaming.md](07-event-streaming.md)** - Native LangGraph streaming
- **[08-configuration.md](08-configuration.md)** - Configuration preservation

### Integration & UI

- **[09-tui-integration.md](09-tui-integration.md)** - Textual UI connection
- **[10-testing.md](10-testing.md)** - Test strategy and validation

## Timeline

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

**Total Timeline: 9-11 weeks**

## Next Steps

1. **Create Documents**: Generate the full documentation files in this directory
2. **Review & Approval**: Review this plan with stakeholders
3. **Phase 1 Kickoff**: Start with dependency integration
4. **Weekly Checkpoints**: Review progress at each phase boundary
5. **User Testing**: Begin after Phase 2 (Engine implementation)

## Questions

Before proceeding with implementation, confirm:

1. **Python Version**: Still targeting 3.12+ (required for modern LangChain)
2. **LangChain Version**: Using LangChain 1.2.0 (latest stable)
3. **Documentation**: Full detailed documents created in this directory
4. **Testing**: Tests created separately from documentation
5. **User Testing**: Begin after Phase 2 completion

---

**Migration Lead**: AI Assistant  
**Timeline**: 9-11 weeks total  
**Risk Level**: Medium (gradual rollout with feature flags)  
**Impact**: Major architectural improvement with simplified codebase

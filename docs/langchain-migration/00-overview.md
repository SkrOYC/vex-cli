# 00 - Overview

## Vision

Transform Mistral Vibe from a DeepAgents-dependent application to a direct LangChain 1.2.0 integration using `create_agent()` and the new middleware API. The result will be a significantly simplified codebase with full architectural control while preserving the excellent Textual TUI experience.

## Guiding Principles

1. **LangChain 1.2.0 as Engine**: Use `create_agent()` and LangGraph as the core agent engine
2. **Preserve TUI**: Keep Mistral Vibe's Textual-based UI as the presentation layer
3. **Simplify Ruthlessly**: Remove adapter layers where native LangChain provides equivalents
4. **Maintain UX**: User experience should be preserved or enhanced, never degraded
5. **Full Customization**: Build middleware stack from scratch without opinionated defaults

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mistral Vibe TUI (Preserved)                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ Chat Input  │ │  Messages   │ │  Approval   │ │  Context  │  │
│  │  Container  │ │   Display   │ │   Dialogs   │ │  Progress │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Thin Adapter Layer (Minimal)                 │
│                  ┌─────────────┐ ┌─────────────┐                │
│                  │   Config    │ │   Tool      │                │
│                  │   Loader    │ │  Adapter    │                │
│                  └─────────────┘ └─────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                LangChain 1.2.0 Engine (Flexible)               │
│          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│          │create_agent │ │ Custom      │ │  HumanInThe │        │
│          │  ()         │ │ Middleware  │ │   Loop      │        │
│          └─────────────┘ └─────────────┘ └─────────────┘        │
│          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│          │  StateGraph │ │ Checkpoint  │ │  Streaming  │        │
│          │  (optional) │ │   Saver     │ │   Events    │        │
│          └─────────────┘ └─────────────┘ └─────────────┘        │
│                          ^ YOUR CHOICE ^                        │
└─────────────────────────────────────────────────────────────────┘
```

## What Gets Removed (Simplified)

| Component | Current LOC | Reason for Removal |
|-----------|-------------|-------------------|
| `vibe/core/engine/engine.py` | ~405 lines | Replaced by `VibeLangChainEngine` |
| `vibe/core/engine/adapters.py` | ~256 lines | Replaced by native LangChain patterns |
| `vibe/core/engine/middleware.py` | ~204 lines | Replaced by custom `AgentMiddleware` |
| DeepAgents dependency | - | Full removal, use LangChain 1.2.0 directly |

**Total Reduction: ~865 lines → ~350 lines (60% reduction)**

## What Gets Preserved

| Component | Location | Reason |
|-----------|----------|--------|
| Textual TUI | `vibe/cli/textual_ui/` | Core differentiator, excellent UX |
| Configuration System | `vibe/core/config.py` | Project-aware settings |
| Project Detection | `vibe/core/config.py` | Git-based context |
| Update Notifier | `vibe/cli/update_notifier/` | User convenience |
| Autocompletion | `vibe/core/autocompletion/` | Developer productivity |
| Tool System | `vibe/core/tools/` | Works with thin adapter |

## What Gets Enhanced

| Feature | Source | Benefit |
|---------|--------|---------|
| Middleware Stack | Custom `AgentMiddleware` | Full control, no forced defaults |
| State Management | TypedDict + Checkpointers | Proper schema with persistence |
| Event Streaming | Native LangGraph | Multiple modes, direct access |
| Approval Flow | Native `HumanInTheLoopMiddleware` | Simplified, no adapter |
| Token Tracking | Native `usage_metadata` | Accurate, no estimation |

## Migration Phases

1. **Phase 1: Foundation** - Add LangChain 1.2.0, create engine abstraction ✅ Complete
2. **Phase 2: Agent Engine** - Replace `create_deep_agent()` with `create_agent()` ✅ Complete
3. **Phase 3: Middleware** - Build custom middleware stack ✅ Complete
4. **Phase 4: Tools** - Migrate tool adapter for native LangChain ✅ Complete
5. **Phase 5: DeepAgents Removal** - Remove legacy DeepAgents engine files ✅ Complete
6. **Phase 6: Approval** - Use native HITL middleware ✅ Complete
7. **Phase 7: Events** - Native LangGraph streaming ✅ Complete
8. **Phase 8: Configuration** - Remove DeepAgents config flags ✅ Complete
9. **Phase 9: Testing** - Comprehensive validation ✅ Complete
10. **Phase 10: Cleanup** - Final verification and documentation ✅ Complete

## Success Criteria

- [x] All existing TUI features work unchanged
- [x] All existing tools function correctly
- [x] Configuration system preserved
- [x] Project detection works
- [x] New features (custom middleware, better streaming) accessible via TUI
- [x] Test suite passes
- [x] Performance meets or exceeds original
- [x] No DeepAgents imports remain in codebase
- [x] Adapter layers completely removed

## Document Index

- `01-dependencies.md` - Package dependencies and compatibility
- `02-agent-engine.md` - Core agent replacement strategy
- `03-tools.md` - Tool migration approach
- `04-middleware.md` - Middleware implementation plan
- `05-state-management.md` - State schema and checkpointing
- `06-approval-system.md` - Native HITL integration
- `07-event-streaming.md` - Native LangGraph streaming
- `08-configuration.md` - Configuration preservation
- `09-tui-integration.md` - Textual UI connection
- `10-testing.md` - Test strategy and validation

# DeepAgents Migration Plan: Overview

## Vision

Transform Mistral Vibe into a DeepAgents-first application while preserving its rich Textual TUI. The result will be a significantly simplified codebase with enhanced capabilities (planning, subagents, advanced middleware) powered by the DeepAgents library.

## Guiding Principles

1. **DeepAgents as Engine**: Use `create_deep_agent()` and LangGraph as the core agent engine
2. **Preserve TUI**: Keep Mistral Vibe's Textual-based UI as the presentation layer
3. **Simplify Ruthlessly**: Remove custom implementations where DeepAgents provides equivalents
4. **Maintain UX**: User experience should be preserved or enhanced, never degraded

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
│                    Thin Adapter Layer (New)                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   Event     │ │  Approval   │ │   Config    │               │
│  │  Translator │ │   Bridge    │ │   Loader    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DeepAgents Engine (New)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ create_deep │ │ Filesystem  │ │  SubAgent   │               │
│  │   _agent()  │ │  Backend    │ │ Middleware  │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  TodoList   │ │Summarization│ │    HITL     │               │
│  │ Middleware  │ │ Middleware  │ │ Middleware  │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

## What Gets Removed (Simplified)

| Component | Current LOC | Reason for Removal |
|-----------|-------------|-------------------|
| `vibe/core/agent.py` | ~960 lines | Replaced by `create_deep_agent()` |
| `vibe/core/middleware.py` | ~170 lines | Replaced by DeepAgents middleware |
| `vibe/core/tools/base.py` | ~284 lines | Simplified to thin wrappers |
| `vibe/core/tools/manager.py` | ~260 lines | Simplified tool registration |
| `vibe/core/llm/backend/*.py` | ~400 lines | Replaced by BaseChatModel |

**Total Reduction: ~2000 lines → ~400 lines (80% reduction)**

## What Gets Preserved

| Component | Location | Reason |
|-----------|----------|--------|
| Textual TUI | `vibe/cli/textual_ui/` | Core differentiator, excellent UX |
| Configuration System | `vibe/core/config.py` | Project-aware settings |
| Project Detection | `vibe/core/config.py` | Git-based context |
| Update Notifier | `vibe/cli/update_notifier/` | User convenience |
| Autocompletion | `vibe/core/autocompletion/` | Developer productivity |

## What Gets Added (Enhanced)

| Feature | Source | Benefit |
|---------|--------|---------|
| Subagent Delegation | DeepAgents SubAgentMiddleware | Parallel task execution |
| Advanced Planning | DeepAgents TodoListMiddleware | Structured task management |
| Context Summarization | DeepAgents SummarizationMiddleware | Better long conversations |
| Filesystem Backend | DeepAgents FilesystemBackend | Secure file operations |

## Migration Phases

1. **Phase 1: Foundation** - Add DeepAgents, create adapter layer
2. **Phase 2: Agent Engine** - Replace Agent class with create_deep_agent()
3. **Phase 3: Tools** - Migrate tools to DeepAgents format
4. **Phase 4: Middleware** - Adopt DeepAgents middleware stack
5. **Phase 5: UI Integration** - Connect TUI to new engine
6. **Phase 6: Cleanup** - Remove deprecated code, polish

## Success Criteria

- [ ] All existing TUI features work unchanged
- [ ] All existing tools function correctly
- [ ] Configuration system preserved
- [ ] Project detection works
- [ ] New features (subagents, planning) accessible via TUI
- [ ] Test suite passes
- [ ] Performance meets or exceeds original

## Document Index

- `01-dependencies.md` - Package dependencies and compatibility
- `02-agent-engine.md` - Core agent replacement strategy
- `03-tools.md` - Tool migration approach
- `04-middleware.md` - Middleware adoption plan
- `05-backends.md` - Filesystem and model backends
- `06-approval-system.md` - HITL and permission handling
- `07-event-streaming.md` - Event translation for TUI
- `08-configuration.md` - Config system preservation
- `09-tui-integration.md` - Textual UI connection
- `10-testing.md` - Test strategy and validation

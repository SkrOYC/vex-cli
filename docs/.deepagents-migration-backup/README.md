# DeepAgents Migration Plan

## Executive Summary

This document outlines a comprehensive plan to migrate Mistral Vibe from its custom agent architecture to a DeepAgents-first approach, while preserving the rich Textual TUI that makes Mistral Vibe unique.

## Vision

**Before**: Custom Agent class (960 lines) + complex middleware pipeline + manual conversation loop management

**After**: DeepAgents `create_deep_agent()` (20 lines) + LangGraph orchestration + enhanced capabilities

**Result**: 80% code reduction + advanced features (planning, subagents, better middleware) + preserved UX

## Key Principles

1. **DeepAgents as Engine**: Use `create_deep_agent()` and LangGraph as the core execution engine
2. **Preserve TUI**: Keep Mistral Vibe's Textual UI as the presentation layer - no changes to user experience
3. **Simplify Ruthlessly**: Remove custom implementations wherever DeepAgents provides equivalents
4. **Maintain Compatibility**: All existing configuration, tools, and features continue to work

## Migration Overview

### What Changes
- **Agent Engine**: Replace 960-line custom `Agent` class with DeepAgents `CompiledStateGraph`
- **Middleware**: Replace 6 custom middleware with 7 DeepAgents middleware
- **Tools**: Adapt existing tools to StructuredTool API, gain 3 new filesystem tools
- **Backend**: Replace custom backends with FilesystemBackend + LangChain models
- **Events**: Translate LangGraph events to Vibe TUI events
- **Approval**: Bridge DeepAgents HITL to existing approval dialogs

### What Stays the Same
- **TUI Experience**: Textual interface, approval dialogs, keyboard shortcuts
- **Configuration**: TOML files, project detection, environment variables
- **User Workflows**: Tool usage, conversation flow, project context
- **API Surface**: Configuration loading, CLI commands, plugin system

### New Capabilities Gained
- **Planning**: TodoListMiddleware for structured task management
- **Subagents**: Parallel task delegation with isolated contexts
- **Advanced Context**: SummarizationMiddleware with 170k token triggers
- **Better Middleware**: AnthropicPromptCaching, PatchToolCalls, etc.
- **Enhanced Filesystem**: Security, virtual modes, better error handling

## Implementation Phases

### Phase 1: Foundation & Compatibility (2-3 weeks)
- Add DeepAgents dependency
- Create API bridges (tools, middleware, events)
- Establish engine abstraction layer
- Parallel testing setup

### Phase 2: Core Feature Migration (3-4 weeks)
- Replace Agent class with VibeEngine
- Migrate tools to DeepAgents format
- Adopt DeepAgents middleware stack
- Integrate FilesystemBackend

### Phase 3: Enhanced Features & UI Adaptation (2-3 weeks)
- Add planning and subagent support
- Enhance approval system with HITL
- Update TUI for new event types
- Preserve all existing UI behaviors

### Phase 4: Advanced Capabilities & Polish (2-3 weeks)
- Add context summarization
- Implement prompt caching
- Skills system integration (optional)
- Performance optimization

### Phase 5: Testing & Validation (1-2 weeks)
- Comprehensive test suite
- Feature parity validation
- Performance benchmarking
- User acceptance testing

## Success Metrics

### Code Reduction
- **Core Agent Logic**: 1000+ lines → ~180 lines (82% reduction)
- **Middleware**: 170 lines → 30 lines (80% reduction)
- **Tool Framework**: 284 lines → thin adapters (90% reduction)

### Feature Enhancement
- **Planning**: Basic todos → Structured TodoListMiddleware
- **Delegation**: None → Full subagent support
- **Context Management**: Manual → Automatic summarization
- **Filesystem**: Basic → Advanced with security

### Compatibility
- **Configuration**: 100% backward compatible
- **TUI**: 100% preserved user experience
- **Tools**: All existing tools work identically
- **Performance**: Meets or exceeds original

## Risk Mitigation

### Gradual Rollout
- Feature flags for each component
- Parallel old/new implementation testing
- Ability to rollback any phase

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
- **[01-dependencies.md](01-dependencies.md)** - Package dependencies and compatibility
- **[02-agent-engine.md](02-agent-engine.md)** - Core agent replacement strategy
- **[03-tools.md](03-tools.md)** - Tool migration approach
- **[04-middleware.md](04-middleware.md)** - Middleware adoption plan
- **[05-backends.md](05-backends.md)** - Filesystem and model backends

### Integration & UI
- **[06-approval-system.md](06-approval-system.md)** - HITL and permission handling
- **[07-event-streaming.md](07-event-streaming.md)** - Event translation for TUI
- **[08-configuration.md](08-configuration.md)** - Config system preservation
- **[09-tui-integration.md](09-tui-integration.md)** - Textual UI connection
- **[10-testing.md](10-testing.md)** - Test strategy and validation

## Next Steps

1. **Review & Approval**: Review this plan with stakeholders
2. **Phase 1 Kickoff**: Start with dependency integration
3. **Weekly Checkpoints**: Review progress at each phase boundary
4. **User Testing**: Involve users early in TUI preservation validation

## Contact & Support

For questions about this migration plan:
- Review the detailed documents in this directory
- Check existing Mistral Vibe code for current implementations
- Reference DeepAgents documentation for new capabilities

---

**Migration Lead**: AI Assistant
**Timeline**: 9-13 weeks total
**Risk Level**: Medium (gradual rollout mitigates risks)
**Impact**: Major simplification with feature enhancement

# Phase 2: SummarizationMiddleware Integration

## Issue Type
Enhancement

## Summary
This issue tracks the integration of SummarizationMiddleware configuration in vex-cli, completing the implementation started in Phase 1 (issue #30).

## Phase 1 Status ✅
- [x] Configuration fields added to `VibeConfig`
- [x] Validation implemented for summarization settings
- [x] Unit tests created for configuration
- [x] Documentation updated with current limitations
- [x] Feature documentation created
- [x] DeepAgents enhancement request documented

## Phase 2 Tasks

### Dependencies
- [ ] **DeepAgents Enhancement**: Implement SummarizationMiddleware parameterization
  - PR: [link to DeepAgents PR when created]
  - Issue: [link to DeepAgents issue when created]

### Integration Implementation
- [ ] **Add SummarizationMiddleware to middleware stack**
  - Modify `build_middleware_stack()` in `vibe/core/engine/middleware.py`
  - Conditionally add when `enable_summarization=True`
  - Handle model parameter correctly

- [ ] **Configuration Testing**
  - Test with different trigger/keep combinations
  - Verify no duplicate middleware issues
  - Validate with various model types

### Testing
- [ ] **Integration Tests**
  - Create `tests/integration/test_long_conversation.py`
  - Test 200+ message conversations
  - Verify context preservation
  - Test performance impact

- [ ] **End-to-end Tests**
  - Test with real models (if possible)
  - Verify configuration from TOML files
  - Test CLI integration

### Documentation
- [ ] **Update Migration Guide**
  - Complete `docs/deepagents-migration/04-middleware.md`
  - Add usage examples
  - Document performance implications

- [ ] **Complete Feature Docs**
  - Update `docs/features/summarization.md`
  - Remove "future" language
  - Add troubleshooting examples

## Technical Requirements

### Integration Pattern
```python
# In vibe/core/engine/middleware.py
from langchain.agents.middleware.summarization import SummarizationMiddleware

def build_middleware_stack(
    config: VibeConfig,
    model: "BaseChatModel",  # type: ignore
    backend: "BackendProtocol",  # type: ignore
) -> list[AgentMiddleware]:
    """Build the complete middleware stack for the agent.
    
    Order is important for correct execution:
    1. Subagents (SubAgentMiddleware, optional) - DeepAgents provides TodoList and Filesystem by default
    2. Context warnings (ContextWarningMiddleware, Vibe-specific)
    3. Price limit (PriceLimitMiddleware, Vibe-specific)
    4. Human-in-the-loop (HumanInTheLoopMiddleware, for approvals)
    """
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware

    middleware: list[AgentMiddleware] = []
    
    # DeepAgents provides TodoListMiddleware, FilesystemMiddleware, and SubAgentMiddleware by default
    # Only add custom middleware that's not already provided by DeepAgents
    
    # 1. Subagents (optional, Vibe-specific) - handled by DeepAgents automatically
    # Note: Don't add SubAgentMiddleware manually as it causes duplicate middleware error
    
    # 2. Context warnings (Vibe-specific)
    if config.context_warnings:
        middleware.append(
            ContextWarningMiddleware(
                threshold_percent=0.5, max_context=config.auto_compact_threshold
            )
        )
    
    # 3. Price limit (Vibe-specific)
    if config.max_price is not None:
        # Get pricing from model config
        pricing = {}
        for model_config in config.models:
            pricing[model_config.name] = (
                model_config.input_price / 1_000_000,  # Convert to per-token rate
                model_config.output_price / 1_000_000,
            )
    
        middleware.append(PriceLimitMiddleware(config.max_price, pricing))
    
    # 4. Summarization (if enabled)
    if config.enable_summarization and config.use_deepagents:
        middleware.append(
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", config.summarization_trigger_tokens),
                keep=("messages", config.summarization_keep_messages),
            )
        )
    
    # 5. Human-in-the-loop (for approvals) - independent of price limit
    from vibe.core.engine.permissions import build_interrupt_config
    
    interrupt_on = build_interrupt_config(config)
    if interrupt_on:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    
    return middleware
```

### Validation
- Ensure no duplicate SummarizationMiddleware instances
- Verify DeepAgents defaults are properly overridden
- Test with various model configurations

### Performance Targets
- Additional API cost: +10-15% (expected)
- Latency impact: 2-5 seconds per trigger
- Memory usage: No significant increase

## Acceptance Criteria

- [ ] SummarizationMiddleware integrates without conflicts
- [ ] Configuration options work as documented
- [ ] Long conversations (200+ messages) don't hit context limits
- [ ] Context preservation quality is acceptable
- [ ] Performance impact stays within expected ranges
- [ ] All tests pass
- [ ] Documentation is complete and accurate

## Implementation Notes

### Duplicate Middleware Prevention
Before adding SummarizationMiddleware, verify:
1. DeepAgents provides no way to disable default SummarizationMiddleware
2. Custom middleware runs after defaults (won't replace them)
3. May result in multiple SummarizationMiddleware instances

### Testing Strategy
- Use `model_construct()` to avoid API key requirements in tests
- Mock SummarizationMiddleware for unit tests
- Focus on integration testing for behavior verification

### Rollout Plan
- Feature flag: Use existing `enable_summarization` as toggle
- Conservative defaults: Keep `enable_summarization=False` initially
- Gradual rollout: Monitor community feedback

## Related Issues

- #30 (Phase 1) - SummarizationMiddleware configuration support ✅
- #29 - Performance benchmarks (should include summarization tests)
- DeepAgents enhancement - [link when created]

## Success Metrics

- [ ] 200+ message conversations work reliably
- [ ] Summarization quality maintains context integrity  
- [ ] Performance impact < 15% (cost + latency)
- [ ] User feedback positive for long sessions
- [ ] No regressions in short conversations

---

**Assignee**: TBD  
**Milestone**: v1.2.1 (Phase 2)  
**Priority**: High  
**Status**: Blocked on DeepAgents Enhancement
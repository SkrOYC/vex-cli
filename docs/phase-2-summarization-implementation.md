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
if config.enable_summarization and config.use_deepagents:
    from langchain.agents.middleware.summarization import SummarizationMiddleware
    
    middleware.append(
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", config.summarization_trigger_tokens),
            keep=("messages", config.summarization_keep_messages),
        )
    )
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
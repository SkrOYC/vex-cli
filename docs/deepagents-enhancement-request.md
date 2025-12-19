# DeepAgents SummarizationMiddleware Enhancement Request

## Overview

vex-cli has prepared configuration infrastructure for SummarizationMiddleware customization, but DeepAgents library currently lacks the ability to customize SummarizationMiddleware parameters without causing duplicates.

## Current Limitation

DeepAgents' `create_deep_agent()` function hardcodes SummarizationMiddleware with fixed trigger/keep values:

```python
# In deepagents/graph.py lines 123-128 and 135-140
deepagent_middleware = [
    # ... other middleware ...
    SummarizationMiddleware(
        model=model,
        trigger=trigger,  # Hardcoded: ("fraction", 0.85) or ("tokens", 170000)
        keep=keep,        # Hardcoded: ("fraction", 0.10) or ("messages", 6)
        trim_tokens_to_summarize=None,
    ),
    # ... more middleware ...
]
```

## Proposed Enhancement

### New Parameters for `create_deep_agent()`

```python
def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    # NEW PARAMETERS
    summarization_trigger: ContextSize | None = None,
    summarization_keep: ContextSize | None = None,
    disable_summarization: bool = False,
) -> CompiledStateGraph:
```

### Implementation Details

```python
def create_deep_agent(..., summarization_trigger=None, summarization_keep=None, disable_summarization=False):
    if model is None:
        model = get_default_model()

    # Determine summarization configuration
    if disable_summarization:
        # Don't add SummarizationMiddleware at all
        use_summarization = False
    elif summarization_trigger is not None and summarization_keep is not None:
        # Use user-provided configuration
        use_summarization = True
        trigger = summarization_trigger
        keep = summarization_keep
    else:
        # Use current hardcoded logic (backward compatibility)
        use_summarization = True
        if (
            model.profile is not None
            and isinstance(model.profile, dict)
            and "max_input_tokens" in model.profile
            and isinstance(model.profile["max_input_tokens"], int)
        ):
            trigger = ("fraction", 0.85)
            keep = ("fraction", 0.10)
        else:
            trigger = ("tokens", 170000)
            keep = ("messages", 6)

    # Build middleware stack
    deepagent_middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                # Add summarization to subagents if enabled
                *(SummarizationMiddleware(model=model, trigger=trigger, keep=keep, trim_tokens_to_summarize=None) 
                   if use_summarization else []),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        # Add summarization to main agent if enabled
        *(SummarizationMiddleware(model=model, trigger=trigger, keep=keep, trim_tokens_to_summarize=None) 
           if use_summarization else []),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    
    # ... rest of function unchanged
```

## Benefits

### For vex-cli
- **No duplicate middleware**: Can customize without creating multiple instances
- **User control**: Users can set custom trigger/keep values
- **Backward compatible**: Existing behavior preserved when parameters not specified

### For Other Consumers
- **Flexibility**: All DeepAgents users gain summarization control
- **Optional**: Can disable summarization entirely if needed
- **Customizable**: Fine-tune for specific use cases

## Migration Path

### Phase 1: Add Parameters (Non-breaking)
- Add new optional parameters with `None` defaults
- Maintain current behavior for existing users
- Add documentation for new parameters

### Phase 2: Deprecation Path (Future)
- Document current hardcoded behavior
- Add deprecation warnings for very old usage patterns
- Eventually migrate to parameterized defaults

## Integration Examples

### vex-cli Usage
```python
agent = create_deep_agent(
    model="claude-3-5-sonnet-20241022",
    tools=tools,
    summarization_trigger=("tokens", 180000),
    summarization_keep=("messages", 8),
    # ... other parameters
)
```

### Disable Summarization
```python
agent = create_deep_agent(
    model="claude-3-5-sonnet-20241022", 
    tools=tools,
    disable_summarization=True,
    # ... other parameters
)
```

### Default Behavior (Backward Compatible)
```python
agent = create_deep_agent(
    model="claude-3-5-sonnet-20241022",
    tools=tools,
    # Uses current hardcoded logic automatically
)
```

## Testing Requirements

1. **Backward Compatibility**: Existing tests should pass without changes
2. **Parameter Validation**: Test new parameter combinations
3. **Integration Tests**: Verify middleware stack composition
4. **Performance Tests**: Ensure no regressions

## Alternatives Considered

### Option 1: Middleware Override
- Replace default SummarizationMiddleware entirely when custom parameters provided
- **Pro**: Simple implementation
- **Con**: Breaking change for existing users

### Option 2: Selective Middleware (Chosen)
- Allow users to customize parameters while maintaining defaults
- **Pro**: Backward compatible, flexible
- **Con**: More complex implementation

### Option 3: Separate Factory Function
- Create `create_deep_agent_with_summarization()` function
- **Pro**: Clean separation
- **Con**: Function duplication, API complexity

## Implementation Priority

This enhancement should be prioritized as:
1. **High**: Enables vex-cli SummarizationMiddleware configuration (GitHub issue #30)
2. **Medium**: Benefits all DeepAgents consumers
3. **Low**: No breaking changes required

## Related Issues

- vex-cli #30: Add SummarizationMiddleware for automatic context management
- DeepAgents tracking: [To be created]

## Contact

**Author**: vex-cli maintainers  
**Date**: 2025-12-18  
**Status**: Enhancement Request
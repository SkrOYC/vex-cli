# SummarizationMiddleware Configuration

## Overview

vex-cli provides configuration options for SummarizationMiddleware to enable automatic context management in long conversations. This feature helps prevent context overflow and maintains conversation continuity by summarizing older messages when approaching token limits.

## Current State & Limitations

### DeepAgents Integration

**Important**: vex-cli currently uses DeepAgents library which includes SummarizationMiddleware by default with hardcoded settings. The configuration options below are **prepared for future use** but cannot be fully utilized until DeepAgents supports customization.

### Current DeepAgents Behavior

DeepAgents automatically includes SummarizationMiddleware with these settings:

- **Model-based triggers**: If model has `max_input_tokens` profile:
  - Trigger at 85% of max tokens
  - Keep 10% of context
- **Fallback defaults**: Otherwise:
  - Trigger at 170,000 tokens
  - Keep 6 most recent messages

### Limitation

There is currently **no way to customize** DeepAgents' SummarizationMiddleware settings without causing duplicates. Adding custom SummarizationMiddleware would result in multiple instances running simultaneously.

## Configuration Options

These configuration fields are available in `vibe/config.toml`:

```toml
[deepagents]
# Enable/disable summarization (future use)
enable_summarization = false

# When to trigger summarization (tokens)
summarization_trigger_tokens = 170000

# How many messages to keep after summarization
summarization_keep_messages = 6
```

### Field Descriptions

| Field | Type | Default | Min | Max | Notes |
|-------|------|---------|-----|-----|-------|
| `enable_summarization` | bool | `false` | - | - | Main toggle (future use) |
| `summarization_trigger_tokens` | int | `170000` | `50000` | - | Trigger threshold in tokens |
| `summarization_keep_messages` | int | `6` | `2` | - | Messages to retain |

## Recommended Configurations

### Conservative (Low Cost)
```toml
enable_summarization = true
summarization_trigger_tokens = 180000  # High threshold
summarization_keep_messages = 8         # Keep more context
```

### Balanced (Default)
```toml
enable_summarization = true
summarization_trigger_tokens = 170000  # Standard
summarization_keep_messages = 6         # Standard
```

### Aggressive (Max Context)
```toml
enable_summarization = true
summarization_trigger_tokens = 150000  # Low threshold
summarization_keep_messages = 4         # Minimal context
```

## Current Workarounds

### 1. Context Warnings
Use existing context warnings to monitor usage:

```toml
[vibe]
context_warnings = true
auto_compact_threshold = 200000
```

### 2. Manual Compaction
The system will automatically compact when reaching token limits, maintaining conversation continuity.

### 3. DeepAgents Defaults
Rely on DeepAgents' built-in summarization which works well for most use cases.

## Performance Considerations

### When Fully Implemented
- **Cost Impact**: +10-15% API cost for summarization calls
- **Latency Impact**: 2-5 seconds per summarization trigger
- **Quality**: Depends on model capability and summarization strategy

### Current Status
- **No additional cost** (using DeepAgents defaults)
- **Automatic management** without user configuration
- **Reliable behavior** with proven defaults

## Validation Rules

The configuration includes built-in validation:

- `summarization_trigger_tokens` must be ≥ 50,000 tokens
- `summarization_keep_messages` must be ≥ 2 messages
- Warning if trigger < `auto_compact_threshold`

## Future Implementation

### Phase 2: Middleware Integration
When DeepAgents supports SummarizationMiddleware customization:

1. **Enable configuration**: Set `enable_summarization = true`
2. **Custom thresholds**: Adjust `summarization_trigger_tokens` and `summarization_keep_messages`
3. **Automatic integration**: No code changes required

### Enhancement Request
vex-cli has submitted enhancement request to DeepAgents for SummarizationMiddleware parameterization:

```python
# Proposed API
create_deep_agent(
    model=model,
    ...,
    summarization_trigger=("tokens", 170000),
    summarization_keep=("messages", 6),
)
```

## Troubleshooting

### Common Issues

**Q: Configuration changes don't take effect**
A: Ensure `use_deepagents = true` is set in config. Summarization only works with DeepAgents backend.

**Q: Getting warning about trigger vs compact threshold**
A: Adjust values so `summarization_trigger_tokens >= auto_compact_threshold` or accept the warning.

**Q: Performance seems slower**
A: Currently using DeepAgents defaults. When custom implementation is available, consider higher trigger thresholds for better performance.

### Error Messages

- `"Summarization trigger must be >= 50000 tokens"`: Increase `summarization_trigger_tokens`
- `"Must keep at least 2 messages"`: Increase `summarization_keep_messages`
- `"Summarization trigger is less than auto compact threshold"`: Consider adjusting values

## Related Documentation

- [DeepAgents Migration: Middleware](../deepagents-migration/04-middleware.md)
- [Configuration Reference](../configuration/)
- [Context Management](../features/context-management.md)
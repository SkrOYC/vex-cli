# 04 - Middleware Migration

## Overview

Replace Mistral Vibe's custom middleware system with DeepAgents' built-in middleware, gaining advanced features while simplifying the codebase.

## Current Middleware System

### `vibe/core/middleware.py` Structure

```python
class ConversationMiddleware(Protocol):
    async def before_turn(self, context: ConversationContext) -> MiddlewareResult: ...
    async def after_turn(self, context: ConversationContext) -> MiddlewareResult: ...
    def reset(self, reset_reason: ResetReason) -> None: ...

class MiddlewarePipeline:
    def add(self, middleware: ConversationMiddleware) -> MiddlewarePipeline: ...
    async def run_before_turn(self, context: ConversationContext) -> MiddlewareResult: ...
    async def run_after_turn(self, context: ConversationContext) -> MiddlewareResult: ...
```

### Current Middleware

| Middleware | Purpose | Lines |
|------------|---------|-------|
| `TurnLimitMiddleware` | Stop after N turns | ~20 |
| `PriceLimitMiddleware` | Stop when cost exceeds limit | ~20 |
| `AutoCompactMiddleware` | Trigger context compaction | ~20 |
| `ContextWarningMiddleware` | Warn at context threshold | ~35 |

**Total: ~170 lines**

## DeepAgents Middleware System

### Built-in Middleware

DeepAgents provides through LangChain:

```python
from langchain.agents.middleware import (
    TodoListMiddleware,        # Planning and task tracking
    HumanInTheLoopMiddleware,  # Approval workflows
    InterruptOnConfig,         # Interrupt configuration
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

from deepagents.middleware import (
    FilesystemMiddleware,      # File operations
    SubAgentMiddleware,        # Task delegation
)
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
```

### Default DeepAgents Stack

```python
# From deepagents/graph.py
deepagent_middleware = [
    TodoListMiddleware(),                    # Planning
    FilesystemMiddleware(backend=backend),   # File ops
    SubAgentMiddleware(...),                 # Delegation
    SummarizationMiddleware(...),            # Context management
    AnthropicPromptCachingMiddleware(...),   # Cost optimization
    PatchToolCallsMiddleware(),              # Error recovery
]
```

## Migration Mapping

### Direct Replacements

| Vibe Middleware | DeepAgents Equivalent | Notes |
|-----------------|----------------------|-------|
| `AutoCompactMiddleware` | `SummarizationMiddleware` | More sophisticated |
| `ContextWarningMiddleware` | Custom middleware | Preserve as custom |
| `TurnLimitMiddleware` | LangGraph config | Via recursion_limit |
| `PriceLimitMiddleware` | Custom middleware | Preserve as custom |

### New Capabilities

| DeepAgents Middleware | Capability | Benefit |
|----------------------|------------|---------|
| `TodoListMiddleware` | Structured planning | Better task organization |
| `SubAgentMiddleware` | Parallel delegation | Faster complex tasks |
| `FilesystemMiddleware` | Secure file ops | Built-in security |
| `PatchToolCallsMiddleware` | Error recovery | More robust execution |

## Implementation Plan

### Phase 1: Custom Middleware Adapters

Create adapters for Vibe-specific middleware:

```python
# vibe/core/engine/middleware.py

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime


class ContextWarningMiddleware(AgentMiddleware):
    """Warn user when context usage reaches threshold.
    
    Preserves Vibe's context warning functionality.
    """

    def __init__(
        self,
        threshold_percent: float = 0.5,
        max_context: int = 200_000,
    ) -> None:
        self.threshold_percent = threshold_percent
        self.max_context = max_context
        self._warned = False

    def before_model(
        self,
        request: ModelRequest,
        state: AgentState,
        runtime: Runtime,
    ) -> ModelRequest | None:
        """Check context usage before model call."""
        if self._warned:
            return None
        
        # Estimate token count from messages
        messages = state.get("messages", [])
        estimated_tokens = sum(
            len(str(m.content)) // 4 for m in messages  # Rough estimate
        )
        
        if estimated_tokens >= self.max_context * self.threshold_percent:
            self._warned = True
            # Inject warning message
            warning = (
                f"⚠️ Context usage at {estimated_tokens:,} tokens "
                f"({estimated_tokens / self.max_context * 100:.0f}% of {self.max_context:,})"
            )
            # Return modified request with warning injected
            # Implementation depends on how we want to surface this
        
        return None


class PriceLimitMiddleware(AgentMiddleware):
    """Stop execution when price limit is exceeded.
    
    Preserves Vibe's cost control functionality.
    """

    def __init__(self, max_price: float) -> None:
        self.max_price = max_price
        self._total_cost = 0.0

    def after_model(
        self,
        response: ModelResponse,
        state: AgentState,
        runtime: Runtime,
    ) -> ModelResponse | None:
        """Track cost after each model call."""
        # Extract usage from response
        usage = getattr(response, "usage_metadata", None)
        if usage:
            # Calculate cost based on token counts
            # This would need model-specific pricing
            input_cost = usage.get("input_tokens", 0) * 0.0000004  # Example rate
            output_cost = usage.get("output_tokens", 0) * 0.000002
            self._total_cost += input_cost + output_cost
        
        if self._total_cost > self.max_price:
            # Raise interrupt or modify response
            raise RuntimeError(
                f"Price limit exceeded: ${self._total_cost:.4f} > ${self.max_price:.2f}"
            )
        
        return None
```

### Phase 2: Middleware Stack Configuration

Configure the complete middleware stack:

```python
# vibe/core/engine/engine.py

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    TodoListMiddleware,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware

from deepagents.middleware import FilesystemMiddleware, SubAgentMiddleware

from vibe.core.engine.middleware import (
    ContextWarningMiddleware,
    PriceLimitMiddleware,
)


def build_middleware_stack(
    config: VibeConfig,
    model: BaseChatModel,
    backend: BackendProtocol,
) -> list[AgentMiddleware]:
    """Build the middleware stack for the agent."""
    
    middleware = []
    
    # Planning (from DeepAgents)
    middleware.append(TodoListMiddleware())
    
    # Filesystem (from DeepAgents)
    middleware.append(FilesystemMiddleware(backend=backend))
    
    # Subagents (from DeepAgents) - optional based on config
    if config.enable_subagents:
        middleware.append(
            SubAgentMiddleware(
                default_model=model,
                default_tools=[],  # Will inherit from main agent
                general_purpose_agent=True,
            )
        )
    
    # Summarization (from DeepAgents)
    middleware.append(
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", config.auto_compact_threshold),
            keep=("messages", 6),
        )
    )
    
    # Context warning (Vibe-specific)
    if config.context_warnings:
        middleware.append(
            ContextWarningMiddleware(
                threshold_percent=0.5,
                max_context=config.auto_compact_threshold,
            )
        )
    
    # Price limit (Vibe-specific) - if configured
    if hasattr(config, "max_price") and config.max_price:
        middleware.append(PriceLimitMiddleware(config.max_price))
    
    # HITL (from DeepAgents)
    interrupt_on = build_interrupt_config(config)
    if interrupt_on:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    
    return middleware
```

### Phase 3: Remove Old Middleware

After migration, remove:

```python
# Delete vibe/core/middleware.py entirely
# The following classes are no longer needed:
# - ConversationMiddleware (Protocol)
# - MiddlewarePipeline
# - TurnLimitMiddleware (use recursion_limit)
# - PriceLimitMiddleware (migrated)
# - AutoCompactMiddleware (replaced by SummarizationMiddleware)
# - ContextWarningMiddleware (migrated)
# - MiddlewareAction (enum)
# - MiddlewareResult (dataclass)
```

## Feature Comparison

### Context Management

**Before (AutoCompactMiddleware):**
```python
class AutoCompactMiddleware:
    async def before_turn(self, context):
        if context.stats.context_tokens >= self.threshold:
            return MiddlewareResult(action=MiddlewareAction.COMPACT)
        return MiddlewareResult()
```

**After (SummarizationMiddleware):**
```python
# Automatic - just configure trigger
SummarizationMiddleware(
    model=model,
    trigger=("tokens", 170000),    # or ("fraction", 0.85)
    keep=("messages", 6),          # or ("fraction", 0.10)
)
```

### Planning

**Before (Basic Todo Tool):**
```python
# Manual tool calls, no structured state
class Todo(BaseTool):
    async def run(self, args: TodoArgs) -> TodoResult:
        if args.action == "write":
            self.state.todos = args.todos
        return TodoResult(todos=self.state.todos)
```

**After (TodoListMiddleware):**
```python
# Automatic state management, structured planning
TodoListMiddleware()
# Provides write_todos and read_todos tools
# State automatically persisted in LangGraph
```

### Subagent Delegation

**Before:** Not available

**After (SubAgentMiddleware):**
```python
SubAgentMiddleware(
    default_model=model,
    default_tools=tools,
    subagents=[
        {
            "name": "researcher",
            "description": "Research complex topics",
            "system_prompt": "You are a research assistant...",
        }
    ],
    general_purpose_agent=True,  # Auto-creates general agent
)
```

## Validation Checklist

- [ ] TodoListMiddleware provides todo tools
- [ ] FilesystemMiddleware provides file tools
- [ ] SubAgentMiddleware enables delegation
- [ ] SummarizationMiddleware handles context (with limitations)
- [ ] Context warnings display in TUI
- [ ] Price limits stop execution
- [ ] HITL interrupts work with TUI approval
- [ ] Middleware stack executes in correct order
- [ ] No regressions from old middleware

## SummarizationMiddleware Limitations

### Current State (v1.2.0)
- **DeepAgents provides SummarizationMiddleware by default** with hardcoded settings
- **No customization available** - trigger/keep values are hardcoded (85%/10% or 170k/6 messages)
- **Duplicate middleware instances**: SummarizationMiddleware appears in both main agent and subagent stacks
- **vex-cli configuration fields are ready** but cannot be used until DeepAgents supports customization

### Configuration Support (Ready for Future)
vex-cli has added configuration support for future DeepAgents customization:

```python
# In vibe/core/config.py
class VibeConfig(BaseSettings):
    # Summarization settings (for future DeepAgents customization)
    enable_summarization: bool = Field(default=False)
    summarization_trigger_tokens: int = Field(default=170000)
    summarization_keep_messages: int = Field(default=6)
```

### Current Recommendations
1. **Use DeepAgents defaults** - They provide sensible automatic summarization
2. **Monitor context usage** with existing `ContextWarningMiddleware`
3. **Manual compaction** via `auto_compact_threshold` when needed
4. **Configuration fields are ready** for when DeepAgents supports customization

### Future Enhancement Needed
DeepAgents library needs to expose SummarizationMiddleware parameters in `create_deep_agent()`:

```python
# Proposed enhancement
def create_deep_agent(
    ...,
    summarization_trigger: ContextSize | None = None,
    summarization_keep: ContextSize | None = None,
    ...
):
```

Until then, vex-cli configuration cannot be fully utilized without causing duplicate middleware.

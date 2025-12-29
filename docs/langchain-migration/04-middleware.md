# 04 - Middleware Migration

## Overview

Replace the current middleware implementation (which works around DeepAgents' forced defaults) with native LangChain 1.2.0 `AgentMiddleware` patterns, building a custom stack that exactly meets Vibe's needs.

## Current Middleware System

### `vibe/core/engine/middleware.py` Structure (204 lines)

```python
"""Custom middleware for DeepAgents integration."""

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

class ContextWarningMiddleware(AgentMiddleware):
    """Warn user when context usage reaches threshold."""
    # Implemented as AgentMiddleware for DeepAgents
    
class PriceLimitMiddleware(AgentMiddleware):
    """Stop execution when price limit is exceeded."""
    # Implemented as AgentMiddleware for DeepAgents

def build_middleware_stack(config, model, backend):
    """Build the middleware stack for DeepAgents.
    
    Note: DeepAgents provides TodoList, Filesystem, and SubAgent middleware
    automatically. We only add Vibe-specific middleware.
    """
    middleware = []
    
    # Context warnings
    if config.context_warnings:
        middleware.append(ContextWarningMiddleware(...))
    
    # Price limit
    if config.max_price is not None:
        middleware.append(PriceLimitMiddleware(...))
    
    # Human-in-the-loop
    interrupt_on = build_interrupt_config(config)
    if interrupt_on:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    
    return middleware
```

**Problems with Current Implementation:**
- Designed to work *with* DeepAgents' forced defaults
- Cannot remove or customize DeepAgents-bundled middleware
- Workaround comments indicating limitations

## New Middleware Implementation

### `vibe/core/engine/langchain_middleware.py` (Target: ~150 lines)

```python
"""LangChain 1.2.0 middleware for Mistral Vibe."""

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
)
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime


class ContextWarningMiddleware(AgentMiddleware):
    """Warn user when context usage reaches threshold.
    
    Preserves Vibe's context warning functionality using native LangChain 1.2.0 patterns.
    """

    def __init__(
        self,
        threshold_percent: float = 0.5,
        max_context: int | None = None,
    ) -> None:
        self.threshold_percent = threshold_percent
        self.max_context = max_context
        self._warned = False

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check context usage before model call and inject warning if needed."""
        if self._warned or self.max_context is None:
            return None

        # Get actual token count from usage metadata if available
        current_tokens = self._get_current_token_count(state)

        if current_tokens >= self.max_context * self.threshold_percent:
            self._warned = True
            warning_message = self._create_warning(current_tokens, self.max_context)
            return {"warning": warning_message}

        return None

    def _get_current_token_count(self, state: AgentState) -> int:
        """Get current token count from usage metadata or estimate from messages."""
        messages = state.get("messages", [])
        
        # Try actual usage metadata first (NEW in LangChain 1.2.0)
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.usage_metadata:
                total_tokens = msg.usage_metadata.get("total_tokens", 0)
                if total_tokens > 0:
                    return total_tokens

        # Fall back to estimation if no usage metadata available
        return self._estimate_tokens(messages)

    def _estimate_tokens(self, messages: list) -> int:
        """Rough token estimation: ~4 characters per token."""
        total_chars = 0
        for msg in messages:
            content = getattr(msg, "content", msg)
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        total_chars += len(part)
                    elif isinstance(part, dict) and "text" in part:
                        total_chars += len(str(part.get("text", "")))
        return total_chars // 4

    def _create_warning(self, current_tokens: int, max_tokens: int) -> str:
        """Create a formatted warning message."""
        percentage_used = (current_tokens / max_tokens) * 100
        return (
            f"You have used {percentage_used:.0f}% of your total context "
            f"({current_tokens:,}/{max_tokens:,} tokens)"
        )


class PriceLimitMiddleware(AgentMiddleware):
    """Stop execution when price limit is exceeded.
    
    Preserves Vibe's cost control functionality using native LangChain 1.2.0 patterns.
    """

    def __init__(
        self,
        max_price: float,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.max_price = max_price
        self.pricing = pricing or {}
        self._total_cost = 0.0

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Track cost after each model call and raise error if limit exceeded."""
        messages = state.get("messages", [])
        
        if messages and isinstance(messages[-1], AIMessage):
            latest_response = messages[-1]
            
            # Native usage_metadata (NEW in LangChain 1.2.0)
            if latest_response.usage_metadata:
                # Get pricing for the model
                model_name = state.get("model_name", "default")
                input_rate, output_rate = self.pricing.get(model_name, (0.0, 0.0))

                # Calculate cost using actual token counts
                input_tokens = latest_response.usage_metadata.get("input_tokens", 0)
                output_tokens = latest_response.usage_metadata.get("output_tokens", 0)
                cost = (input_tokens * input_rate) + (output_tokens * output_rate)
                self._total_cost += cost

                # Check limit
                if self._total_cost > self.max_price:
                    raise RuntimeError(
                        f"Price limit exceeded: ${self._total_cost:.4f} > ${self.max_price:.2f}"
                    )

        return None
```

## Middleware Stack Building

```python
# New: vibe/core/engine/langchain_engine.py (partial)

def _build_middleware_stack(self) -> list[AgentMiddleware]:
    """Build the custom middleware stack for LangChain 1.2.0.
    
    Unlike DeepAgents, we have FULL control over the middleware stack.
    No forced defaults - only what we need.
    """
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    
    middleware: list[AgentMiddleware] = []

    # 1. Context warnings (Vibe-specific)
    if self.config.context_warnings:
        middleware.append(
            ContextWarningMiddleware(
                threshold_percent=0.5,
                max_context=self.config.auto_compact_threshold,
            )
        )

    # 2. Price limit (Vibe-specific)
    if self.config.max_price is not None:
        middleware.append(
            PriceLimitMiddleware(
                max_price=self.config.max_price,
                pricing=self._get_pricing_config(),
            )
        )

    # 3. Human-in-the-loop (native LangChain 1.2.0)
    # No ApprovalBridge needed - native integration!
    interrupt_on = self._build_interrupt_config()
    if interrupt_on:
        middleware.append(
            HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
        )

    return middleware
```

## Comparison with DeepAgents

| Aspect | DeepAgents (Before) | LangChain 1.2.0 (After) |
|--------|---------------------|--------------------------|
| **Middleware Base** | `AgentMiddleware` | `AgentMiddleware` (same) |
| **Planning** | Forced `TodoListMiddleware` | Optional (add if needed) |
| **Filesystem** | Forced `FilesystemMiddleware` | Optional (add if needed) |
| **Subagents** | Forced `SubAgentMiddleware` | Optional (add if needed) |
| **Summarization** | Forced `SummarizationMiddleware` | Optional (add if needed) |
| **HITL** | Via adapter (`ApprovalBridge`) | Native `HumanInTheLoopMiddleware` |
| **Custom Middleware** | Additive only | Full control (add/remove/reorder) |
| **Token Tracking** | Manual estimation | Native `usage_metadata` |

## Validation Checklist

- [ ] Context warnings display in TUI
- [ ] Price limits stop execution
- [ ] HITL interrupts work with native middleware
- [ ] No forced middleware from DeepAgents
- [ ] Middleware stack executes in correct order
- [ ] Token counts are accurate
- [ ] No regressions from old middleware

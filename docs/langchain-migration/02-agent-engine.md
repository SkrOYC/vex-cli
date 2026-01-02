# 02 - Agent Engine Replacement

## Overview

Replace the current `VibeEngine` (using `create_deep_agent()`) with a new `VibeLangChainEngine` using LangChain 1.2.0's `create_agent()` function, reducing complexity while gaining full architectural control.

## Current Implementation Analysis

### `vibe/core/engine/engine.py` Structure (405 lines)

```python
"""DeepAgents-powered agent engine for Mistral Vibe."""

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from vibe.core.config import VibeConfig
from vibe.core.engine.adapters import ApprovalBridge, EventTranslator  # ADAPTERS
from vibe.core.engine.middleware import build_middleware_stack
from vibe.core.engine.tools import VibeToolAdapter

class VibeEngine:
    """Thin wrapper around DeepAgents for TUI integration."""

    def __init__(self, config: VibeConfig):
        self.config = config
        self.approval_bridge: ApprovalBridge | None = None  # ADAPTER
        self.event_translator = EventTranslator(config)     # ADAPTER
        # ...

    def _create_backend(self) -> Any:
        return FilesystemBackend(
            root_dir=self.config.effective_workdir, 
            virtual_mode=False  # Hardcoded!
        )

    def initialize(self) -> None:
        model = self._create_model()
        tools = VibeToolAdapter.get_all_tools(self.config)
        self._agent = create_deep_agent(  # DeepAgents call
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            backend=backend,  # DeepAgents-specific
            middleware=middleware,
            checkpointer=self._checkpointer,
        )
```

**Problems with Current Implementation:**
- Forced DeepAgents middleware defaults (TodoList, Filesystem, SubAgent)
- `EventTranslator` adapter layer for event processing
- `ApprovalBridge` adapter for interrupt handling
- Manual token estimation (`_estimate_tokens` method)
- Rough compaction implementation
- Hardcoded `virtual_mode=False`

## New Implementation

### `vibe/core/engine/langchain_engine.py` (Target: ~200 lines)

```python
"""LangChain 1.2.0 powered agent engine for Mistral Vibe."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_middleware import (
    ContextWarningMiddleware,
    PriceLimitMiddleware,
)
from vibe.core.engine.tools import VibeToolAdapter


class AgentState(TypedDict):
    """Custom state schema for the Vibe agent."""
    messages: list[Any]  # Chat messages
    context_tokens: int  # Token count tracking
    warning: str | None  # Context warning message


class VibeLangChainEngine:
"""LangChain 1.2.0 powered agent engine for Mistral Vibe."""

    def __init__(
        self,
        config: VibeConfig,
    ) -> None:
        self.config = config
        self._agent: CompiledStateGraph | None = None
        self._checkpointer = InMemorySaver()
        self._thread_id = "vibe-session"
        self._stats = VibeEngineStats()

    def _create_model(self) -> BaseChatModel:
        """Create LangChain model from Vibe config."""
        from vibe.core.engine.models import create_model_from_config
        return create_model_from_config(self.config)

    def _build_interrupt_config(self) -> dict[str, Any]:
        """Build HITL interrupt config from Vibe tool permissions."""
        from vibe.core.engine.permissions import build_interrupt_config
        return build_interrupt_config(self.config)

    def _get_system_prompt(self) -> str:
        """Build system prompt from Vibe config."""
        from vibe.core.system_prompt import get_universal_system_prompt
        from vibe.core.tools.manager import ToolManager

        tool_manager = ToolManager(self.config)
        return get_universal_system_prompt(tool_manager, self.config)

    def _build_middleware_stack(self) -> list[AgentMiddleware]:
        """Build the custom middleware stack for LangChain 1.2.0."""
        middleware: list[AgentMiddleware] = []

        # Context warnings (Vibe-specific)
        if self.config.context_warnings:
            middleware.append(
                ContextWarningMiddleware(
                    threshold_percent=0.5,
                    max_context=self.config.auto_compact_threshold,
                )
            )

        # Price limit (Vibe-specific)
        if self.config.max_price is not None:
            middleware.append(
                PriceLimitMiddleware(
                    max_price=self.config.max_price,
                    pricing=self._get_pricing_config(),
                )
            )

        # Human-in-the-loop (native LangChain 1.2.0)
        interrupt_on = self._build_interrupt_config()
        if interrupt_on:
            middleware.append(
                HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
            )

        return middleware

    def initialize(self) -> None:
        """Initialize the LangChain 1.2.0 agent."""
        model = self._create_model()
        tools = VibeToolAdapter.get_all_tools(self.config)

        # LangChain 1.2.0 create_agent() - NO DeepAgents!
        self._agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            middleware=self._build_middleware_stack(),
            checkpointer=self._checkpointer,
            # CRITICAL: interrupt_before=["tools"] CONFLICTS WITH HITL MIDDLEWARE
            # DELETE THIS LINE - See docs/langchain-migration/06-approval-system.md
        )

    async def run(self, user_message: str) -> AsyncGenerator[Any, None]:
        """Run a conversation turn, yielding native LangGraph events."""
        if self._agent is None:
            self.initialize()

        config: RunnableConfig = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": self.config.max_recursion_depth,
        }

        # Stream native LangGraph events - NO EventTranslator!
        async for event in self._agent.astream_events(
            {"messages": [("user", user_message)]},
            config=config,
            version="v2",
        ):
            yield event

    async def handle_approval(
        self, approved: bool, request_id: str, feedback: str | None = None
    ) -> None:
        """Handle approval decision from TUI."""
        # Native LangGraph resume - NO ApprovalBridge!
        from langgraph.types import Command

        config = {"configurable": {"thread_id": self._thread_id}}

        if approved:
            await self._agent.ainvoke(
                Command(resume={"approved": True, "feedback": feedback}),
                config=config,
            )
        else:
            await self._agent.ainvoke(
                Command(resume={"approved": False, "feedback": feedback}),
                config=config,
            )

    def reset(self) -> None:
        """Reset conversation state."""
        self._checkpointer = InMemorySaver()
        self._thread_id = f"vibe-session-{uuid4()}"
        self._agent = None

    @property
    def stats(self) -> "VibeEngineStats":
        """Get current session statistics."""
        if self._agent is not None:
            state = self._agent.get_state(
                {"configurable": {"thread_id": self._thread_id}}
            )
            messages = state.values.get("messages", [])
            context_tokens = self._get_actual_token_count(messages)
            self._stats.context_tokens = context_tokens
            self._stats._messages = len(messages)

        return self._stats

    def _get_actual_token_count(self, messages: list) -> int:
        """Get actual token count from usage metadata (no estimation!)."""
        total_tokens = 0
        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                total_tokens += msg.usage_metadata.get("input_tokens", 0)
                total_tokens += msg.usage_metadata.get("output_tokens", 0)
        return total_tokens
```

## Migration Steps

### Step 1: Create New Engine Module

```
vibe/core/engine/
├── __init__.py              # Update exports
├── engine.py                # KEEP temporarily for comparison
├── langchain_engine.py      # NEW: VibeLangChainEngine
├── langchain_middleware.py  # NEW: Custom middleware
├── tools.py                 # UPDATE: VibeToolAdapter for LangChain
├── models.py                # No changes needed
├── permissions.py           # No changes needed
└── adapters.py              # REMOVE after migration
```

### Step 2: Parallel Implementation

```python
# Feature flag for gradual rollout
USE_LANGCHAIN = os.getenv("VIBE_USE_LANGCHAIN", "true").lower() == "true"

if USE_LANGCHAIN:
    from vibe.core.engine.langchain_engine import VibeLangChainEngine as VibeEngine
else:
    from vibe.core.engine.engine import VibeEngine  # Legacy
```

### Step 3: TUI Integration

```python
# vibe/cli/textual_ui/app.py (minimal changes)

class VibeApp(App):
    def __init__(self, config: VibeConfig) -> None:
        super().__init__()
        self.config = config
        self.engine = VibeLangChainEngine(  # Changed class name only
            config=config,
        )

    async def on_user_message(self, message: str) -> None:
        # Native events - no EventTranslator!
        async for event in self.engine.run(message):
            await self._handle_event(event)
```

### Step 4: Remove Old Engine

Once migration is validated:
- Delete `vibe/core/engine/engine.py`
- Remove `vibe/core/engine/adapters.py`
- Update `vibe/core/engine/__init__.py`
- Update all imports

## Comparison

### Before (DeepAgents Engine)

```python
# 405 lines with adapter overhead
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

class VibeEngine:
    def __init__(self, config):
        self.approval_bridge = ApprovalBridge(...)  # Adapter
        self.event_translator = EventTranslator(config)  # Adapter
        # ...

    def _estimate_tokens(self, messages):  # Manual estimation!
        total_tokens = 0
        for msg in messages:
            # Character-based approximation
            total_tokens += len(str(msg.content)) // 4
        return total_tokens
```

### After (LangChain 1.2.0 Engine)

```python
# ~200 lines, no adapters
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

class VibeLangChainEngine:
    def __init__(self, config):
        # No adapters needed!
        # ...

    def _get_actual_token_count(self, messages):  # Native!
        total_tokens = 0
        for msg in messages:
            if msg.usage_metadata:  # Actual metadata!
                total_tokens += msg.usage_metadata["input_tokens"]
                total_tokens += msg.usage_metadata["output_tokens"]
        return total_tokens
```

## Features Comparison

| Feature | Before (DeepAgents) | After (LangChain 1.2.0) |
|---------|---------------------|--------------------------|
| **Agent Creation** | `create_deep_agent()` | `create_agent()` |
| **Middleware** | Bundled, with workarounds | Custom `AgentMiddleware` |
| **State Schema** | Implicit | Explicit `TypedDict` |
| **Token Tracking** | Manual estimation (`// 4`) | Native `usage_metadata` |
| **Event Streaming** | Translated via `EventTranslator` | Native LangGraph events |
| **Approval Flow** | `ApprovalBridge` adapter | Native `HumanInTheLoopMiddleware` |
| **Checkpoints** | Basic InMemorySaver | Full LangGraph checkpointing |
| **Interrupt Handling** | Custom resume/reject logic | Native `Command(resume=...)` |

## Validation Checklist

- [ ] Engine initializes successfully
- [ ] Model connects to provider API
- [ ] Tools execute correctly
- [ ] Streaming events reach TUI
- [ ] Approval workflow functions (native HITL)
- [ ] State persists across turns
- [ ] Reset clears conversation
- [ ] Token counts are accurate (no estimation)
- [ ] Performance acceptable
- [ ] No adapter layer code remains

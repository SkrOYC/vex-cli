# 02 - Agent Engine Replacement

## Overview

Replace the custom 960-line `Agent` class with DeepAgents' `create_deep_agent()`, reducing complexity by ~90% while gaining advanced features.

## Current Implementation Analysis

### `vibe/core/agent.py` Structure

```python
class Agent:
    def __init__(self, config, auto_approve, message_observer, ...):
        # 50+ lines of initialization
        self.tool_manager = ToolManager(config)
        self.middleware_pipeline = MiddlewarePipeline()
        self.messages = []
        self.stats = AgentStats()
        ...

    async def _conversation_loop(self, user_msg):
        # 100+ lines of complex loop management
        while not should_break_loop:
            result = await self.middleware_pipeline.run_before_turn(...)
            async for event in self._handle_middleware_result(result):
                yield event
            async for event in self._perform_llm_turn():
                yield event
            ...

    async def _perform_llm_turn(self):
        # 150+ lines of LLM interaction
        ...

    async def _execute_tools(self, tool_calls):
        # 100+ lines of tool execution
        ...
```

**Problems:**
- Complex state machine manually implemented
- Tight coupling between components
- Error-prone conversation loop
- Manual streaming management
- Custom middleware orchestration

## New Implementation

### `vibe/core/engine.py` (New File)

```python
"""DeepAgents-powered agent engine for Mistral Vibe."""

from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    TodoListMiddleware,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from vibe.core.config import VibeConfig
from vibe.core.engine.adapters import (
    ApprovalBridge,
    EventTranslator,
    VibeToolAdapter,
)


class VibeEngine:
    """Thin wrapper around DeepAgents for TUI integration."""

    def __init__(
        self,
        config: VibeConfig,
        approval_callback: ApprovalBridge | None = None,
    ) -> None:
        self.config = config
        self.approval_bridge = approval_callback
        self._agent: CompiledStateGraph | None = None
        self._checkpointer = InMemorySaver()
        self._thread_id = "vibe-session"

    def _create_model(self) -> ChatMistralAI:
        """Create LangChain model from Vibe config."""
        model_config = self.config.get_model_config()
        provider_config = self.config.get_provider_config(model_config.provider)
        
        return ChatMistralAI(
            model=model_config.name,
            api_key=provider_config.get_api_key(),
            temperature=model_config.temperature,
            max_tokens=16384,
        )

    def _build_interrupt_config(self) -> dict[str, bool | InterruptOnConfig]:
        """Build HITL interrupt config from Vibe tool permissions."""
        interrupt_on = {}
        
        # Map Vibe permissions to DeepAgents interrupts
        for tool_name, tool_config in self.config.tools.items():
            if tool_config.permission == "ask":
                interrupt_on[tool_name] = True
            elif tool_config.permission == "never":
                interrupt_on[tool_name] = InterruptOnConfig(
                    before=True,
                    # Will be rejected by approval bridge
                )
        
        return interrupt_on

    def _get_system_prompt(self) -> str:
        """Build system prompt from Vibe config."""
        from vibe.core.system_prompt import get_universal_system_prompt
        return get_universal_system_prompt(self.config)

    def initialize(self) -> None:
        """Initialize the DeepAgents engine."""
        model = self._create_model()
        
        # Get tools adapted from Vibe format
        tools = VibeToolAdapter.get_all_tools(self.config)
        
        # Build the agent
        self._agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            backend=FilesystemBackend(
                root_dir=self.config.effective_workdir,
                virtual_mode=False,
            ),
            interrupt_on=self._build_interrupt_config(),
            checkpointer=self._checkpointer,
        )

    async def run(self, user_message: str) -> AsyncGenerator[Any, None]:
        """Run a conversation turn, yielding events for the TUI."""
        if self._agent is None:
            self.initialize()

        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": 1000,
        }

        # Stream events from the agent
        async for event in self._agent.astream_events(
            {"messages": [("user", user_message)]},
            config=config,
            version="v2",
        ):
            # Translate DeepAgents events to Vibe TUI events
            translated = EventTranslator.translate(event)
            if translated is not None:
                yield translated

    async def handle_approval(self, approved: bool, feedback: str | None = None) -> None:
        """Handle approval decision from TUI."""
        if self.approval_bridge:
            await self.approval_bridge.respond(approved, feedback)

    def reset(self) -> None:
        """Reset conversation state."""
        self._checkpointer = InMemorySaver()
        self._thread_id = f"vibe-session-{uuid4()}"

    @property
    def stats(self) -> dict[str, Any]:
        """Get current session statistics."""
        # Extract from LangGraph state
        state = self._agent.get_state(
            {"configurable": {"thread_id": self._thread_id}}
        )
        return {
            "messages": len(state.values.get("messages", [])),
            "todos": state.values.get("todos", []),
        }
```

## Migration Path

### Step 1: Create Engine Module

```
vibe/core/engine/
├── __init__.py
├── engine.py        # VibeEngine class
├── adapters.py      # Event translation, approval bridge
└── tools.py         # Tool adaptation utilities
```

### Step 2: Parallel Implementation

Keep existing `Agent` class while building `VibeEngine`. Both can coexist.

```python
# Feature flag for gradual rollout
USE_DEEPAGENTS = os.getenv("VIBE_USE_DEEPAGENTS", "false").lower() == "true"

if USE_DEEPAGENTS:
    from vibe.core.engine import VibeEngine as AgentClass
else:
    from vibe.core.agent import Agent as AgentClass
```

### Step 3: TUI Integration

Update `VibeApp` to use `VibeEngine`:

```python
# vibe/cli/textual_ui/app.py

class VibeApp(App):
    def __init__(self, config: VibeConfig) -> None:
        super().__init__()
        self.config = config
        self.engine = VibeEngine(
            config=config,
            approval_callback=self._create_approval_bridge(),
        )

    async def on_user_message(self, message: str) -> None:
        async for event in self.engine.run(message):
            await self._handle_event(event)
```

### Step 4: Remove Old Agent

Once migration is validated, remove:
- `vibe/core/agent.py`
- Related imports and references

## Comparison

### Before (Custom Agent)

```python
# 50+ lines just for initialization
class Agent:
    def __init__(self, config, auto_approve, message_observer, max_turns, 
                 max_price, backend, enable_streaming):
        self.config = config
        self.tool_manager = ToolManager(config)
        self.messages: list[LLMMessage] = []
        self.stats = AgentStats()
        self._new_messages: list[LLMMessage] = []
        self._last_chunk: LLMChunk | None = None
        self._approval_callback = None
        self._approval_event = asyncio.Event()
        self._approval_response = None
        self.auto_approve = auto_approve
        self.message_observer = message_observer
        self.enable_streaming = enable_streaming
        
        # Complex middleware setup
        self.middleware_pipeline = MiddlewarePipeline()
        if max_turns:
            self.middleware_pipeline.add(TurnLimitMiddleware(max_turns))
        if max_price:
            self.middleware_pipeline.add(PriceLimitMiddleware(max_price))
        # ... more middleware
```

### After (DeepAgents Engine)

```python
# 10 lines for equivalent functionality
class VibeEngine:
    def __init__(self, config: VibeConfig, approval_callback=None):
        self.config = config
        self.approval_bridge = approval_callback
        self._agent = None
        self._checkpointer = InMemorySaver()

    def initialize(self):
        self._agent = create_deep_agent(
            model=self._create_model(),
            tools=VibeToolAdapter.get_all_tools(self.config),
            system_prompt=self._get_system_prompt(),
            backend=FilesystemBackend(root_dir=self.config.effective_workdir),
            interrupt_on=self._build_interrupt_config(),
            checkpointer=self._checkpointer,
        )
```

## Features Gained

| Feature | Before | After |
|---------|--------|-------|
| Planning (Todos) | Basic tool | TodoListMiddleware with structured state |
| Subagents | Not available | SubAgentMiddleware for parallel tasks |
| Summarization | Manual compact | SummarizationMiddleware automatic |
| State Persistence | Custom implementation | LangGraph checkpointer |
| Streaming | Manual async generator | Built-in astream_events |

## Validation Checklist

- [ ] Engine initializes successfully
- [ ] Model connects to Mistral API
- [ ] Tools execute correctly
- [ ] Streaming events reach TUI
- [ ] Approval workflow functions
- [ ] State persists across turns
- [ ] Reset clears conversation
- [ ] Performance acceptable

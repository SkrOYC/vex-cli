"""DeepAgents-powered agent engine for Mistral Vibe."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from typing import Any
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from vibe.core.config import VibeConfig
from vibe.core.engine.adapters import ApprovalBridge, EventTranslator
from vibe.core.engine.models import create_model_from_config
from vibe.core.engine.tools import VibeToolAdapter
from vibe.core.types import BaseEvent


class VibeEngine:
    """Thin wrapper around DeepAgents for TUI integration."""

    def __init__(
        self, config: VibeConfig, approval_callback: ApprovalBridge | None = None
    ) -> None:
        self.config = config
        self.approval_bridge = approval_callback
        self.event_translator = EventTranslator(config)
        self._agent: CompiledStateGraph | None = None
        self._checkpointer = InMemorySaver()
        self._thread_id = "vibe-session"

        # Check if DeepAgents should be used based on the feature flag
        self._use_deepagents = config.use_deepagents

    def _create_model(self) -> Any:
        """Create LangChain model from Vibe config."""
        return create_model_from_config(self.config)

    def _build_interrupt_config(self) -> dict[str, Any]:
        """Build HITL interrupt config from Vibe tool permissions."""
        interrupt_on = {}

        # Map Vibe permissions to DeepAgents interrupts
        for tool_name, tool_config in self.config.tools.items():
            if tool_config.permission.name == "ASK":
                interrupt_on[tool_name] = True
            elif tool_config.permission.name == "NEVER":
                interrupt_on[tool_name] = {"before": True}  # Placeholder

        return interrupt_on

    def _get_system_prompt(self) -> str:
        """Build system prompt from Vibe config."""
        from vibe.core.system_prompt import get_universal_system_prompt
        from vibe.core.tools.manager import ToolManager

        tool_manager = ToolManager(self.config)
        return get_universal_system_prompt(tool_manager, self.config)

    def initialize(self) -> None:
        """Initialize the DeepAgents engine."""
        model = self._create_model()

        # Get tools adapted from Vibe format
        tools = VibeToolAdapter.get_all_tools(self.config)

        # Build the agent
        self._agent = create_deep_agent(  # type: ignore
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            backend=FilesystemBackend(  # type: ignore
                root_dir=self.config.effective_workdir, virtual_mode=False
            ),
            interrupt_on=self._build_interrupt_config(),
            checkpointer=self._checkpointer,
        )

    async def run(self, user_message: str) -> AsyncGenerator["BaseEvent", None]:
        """Run a conversation turn, yielding events for the TUI."""
        if self._agent is None:
            self.initialize()

        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": 1000,
        }

        # Stream events from the agent - handle the correct event format
        async for event in self._agent.astream_events(  # type: ignore
            {"messages": [("user", user_message)]},
            config=config,
            version="v2",  # type: ignore[attr]
        ):
            # Translate DeepAgents events to Vibe TUI events
            translated = self.event_translator.translate(event.__dict__)  # type: ignore
            if translated is not None:
                yield translated

    async def handle_approval(
        self, approved: bool, feedback: str | None = None
    ) -> None:
        """Handle approval decision from TUI."""
        if self.approval_bridge:
            await self.approval_bridge.respond(approved, feedback)

    def reset(self) -> None:
        """Reset conversation state."""
        self._checkpointer = InMemorySaver()
        self._thread_id = f"vibe-session-{uuid4()}"
        self._agent = None  # Reset the agent so it gets reinitialized on next use

    @property
    def stats(self) -> dict[str, Any]:
        """Get current session statistics."""
        if self._agent is None:
            return {"messages": 0, "todos": [], "context_tokens": 0}
        # Extract from LangGraph state
        state = self._agent.get_state(  # type: ignore
            {"configurable": {"thread_id": self._thread_id}}
        )
        messages = state.values.get("messages", [])
        return {
            "messages": len(messages),
            "todos": state.values.get("todos", []),
            "context_tokens": self._estimate_tokens(messages),
        }

    def _estimate_tokens(self, messages: list) -> int:
        """Estimate token count from messages using actual token metadata if available."""
        total_tokens = 0

        # Try to use actual token usage metadata from messages if available
        for msg in messages:
            # Check if the message has usage metadata (available in newer LangChain versions)
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                usage = msg.usage_metadata
                # Add both input and output tokens if available
                total_tokens += usage.get("input_tokens", 0)
                total_tokens += usage.get("output_tokens", 0)
            # If no usage metadata, fall back to character-based estimation
            elif hasattr(msg, "content"):
                content = msg.content
                if isinstance(content, list):
                    # Handle list of content parts
                    for part in content:
                        if hasattr(part, "text"):
                            total_tokens += len(str(getattr(part, "text", ""))) // 4
                        elif isinstance(part, str):
                            total_tokens += len(part) // 4
                        else:
                            total_tokens += len(str(part)) // 4
                else:
                    total_tokens += len(str(content)) // 4

        return total_tokens

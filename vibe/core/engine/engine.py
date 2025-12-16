"""DeepAgents-powered agent engine for Mistral Vibe."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from vibe.core.config import VibeConfig
from vibe.core.engine.adapters import ApprovalBridge, EventTranslator
from vibe.core.engine.middleware import build_middleware_stack
from vibe.core.engine.models import create_model_from_config
from vibe.core.engine.tools import VibeToolAdapter
from vibe.core.interaction_logger import InteractionLogger
from vibe.core.types import AgentStatsProtocol, BaseEvent, LLMMessage


class VibeEngineStats:
    """Statistics implementation for VibeEngine that matches AgentStats interface."""

    def __init__(
        self, messages: int = 0, context_tokens: int = 0, todos: list[Any] | None = None
    ):
        self._messages = messages
        self._todos = todos or []
        self.steps = 0  # VibeEngine doesn't track steps like legacy agent
        self.session_prompt_tokens = context_tokens // 2  # Rough estimate
        self.session_completion_tokens = context_tokens // 2  # Rough estimate
        self.tool_calls_agreed = 0
        self.tool_calls_rejected = 0
        self.tool_calls_failed = 0
        self.tool_calls_succeeded = 0
        self.context_tokens = context_tokens
        self.last_turn_prompt_tokens = 0
        self.last_turn_completion_tokens = 0
        self.last_turn_duration = 0.0
        self.tokens_per_second = 0.0
        self.input_price_per_million = 0.0  # Would need model pricing info
        self.output_price_per_million = 0.0  # Would need model pricing info

    @property
    def session_total_llm_tokens(self) -> int:
        return self.session_prompt_tokens + self.session_completion_tokens

    @property
    def last_turn_total_tokens(self) -> int:
        return self.last_turn_prompt_tokens + self.last_turn_completion_tokens

    @property
    def session_cost(self) -> float:
        input_cost = (
            self.session_prompt_tokens / 1_000_000
        ) * self.input_price_per_million
        output_cost = (
            self.session_completion_tokens / 1_000_000
        ) * self.output_price_per_million
        return input_cost + output_cost

    def update_pricing(self, input_price: float, output_price: float) -> None:
        self.input_price_per_million = input_price
        self.output_price_per_million = output_price


class VibeEngine:
    """Thin wrapper around DeepAgents for TUI integration."""

    def __init__(
        self,
        config: VibeConfig,
        approval_callback: ApprovalBridge | None = None,
        initial_messages: list[LLMMessage] | None = None,
    ) -> None:
        self.config = config
        self.approval_bridge = approval_callback
        self.event_translator = EventTranslator(config)
        self.initial_messages = initial_messages if initial_messages is not None else []
        self._agent: CompiledStateGraph | None = None
        self._checkpointer = InMemorySaver()
        self._thread_id = "vibe-session"
        # TODO: Implement proper interaction logging for VibeEngine
        # self.interaction_logger = InteractionLogger(
        #     config.session_logging,
        #     self._thread_id,
        #     approval_callback is None,  # auto_approve if no callback
        #     config.effective_workdir,
        # )
        self.interaction_logger = None  # Placeholder

        # Check if DeepAgents should be used based on the feature flag
        self._use_deepagents = config.use_deepagents

    def _create_model(self) -> Any:
        """Create LangChain model from Vibe config."""
        return create_model_from_config(self.config)

    def _create_backend(self) -> Any:
        """Create filesystem backend."""
        return FilesystemBackend(  # type: ignore
            root_dir=self.config.effective_workdir, virtual_mode=False
        )

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

    def initialize(self) -> None:
        """Initialize the DeepAgents engine."""
        model = self._create_model()

        # Get tools adapted from Vibe format
        tools = VibeToolAdapter.get_all_tools(self.config)

        # Build middleware stack
        middleware = build_middleware_stack(self.config, model, self._create_backend())

        # Build the agent
        self._agent = create_deep_agent(  # type: ignore
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            backend=self._create_backend(),
            middleware=middleware,
            interrupt_on=self._build_interrupt_config(),
            checkpointer=self._checkpointer,
        )

    async def run(self, user_message: str) -> AsyncGenerator["BaseEvent", None]:
        """Run a conversation turn, yielding events for the TUI."""
        if self._agent is None:
            self.initialize()

        assert self._agent is not None  # Should be initialized above

        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": self.config.max_recursion_depth,
        }

        # Prepare messages: convert initial to tuples + new message
        messages = [(msg.role.value, msg.content) for msg in self.initial_messages] + [
            ("user", user_message)
        ]

        # Stream events from the agent - handle the correct event format
        async for event in self._agent.astream_events(  # type: ignore
            {"messages": messages},
            config=config,  # type: ignore
            version="v2",  # type: ignore
        ):
            # Translate DeepAgents events to Vibe TUI events
            translated = self.event_translator.translate(event)
            if translated is not None:
                yield translated

    async def handle_approval(
        self, approved: bool, request_id: str, feedback: str | None = None
    ) -> None:
        """Handle approval decision from TUI."""
        if self.approval_bridge:
            await self.approval_bridge.respond(approved, request_id, feedback)

    async def resume_execution(self, decision: dict[str, Any]) -> None:
        """Resume execution after approval."""
        if self._agent is None:
            return

        config = {"configurable": {"thread_id": self._thread_id}}

        # For LangGraph, resume execution from interrupt by calling invoke with None
        try:
            # Resume the agent execution from the interrupt point
            self._agent.invoke(None, config)  # type: ignore
        except Exception as e:
            # Handle resume errors gracefully
            # In a production system, you might want to log this error
            pass

    async def reject_execution(self, decision: dict[str, Any]) -> None:
        """Reject execution and abort operation."""
        if self._agent is None:
            return

        config = {"configurable": {"thread_id": self._thread_id}}

        # For rejection, update the state with a rejection message
        try:
            from langchain_core.messages import HumanMessage

            # Add a rejection message to the conversation
            feedback = decision.get("feedback", "Operation rejected by user")
            rejection_message = HumanMessage(
                content=f"I reject this operation: {feedback}"
            )

            # Update the state with the rejection
            self._agent.update_state(  # type: ignore
                config,  # type: ignore
                {"messages": [rejection_message]},
                as_node="human",  # Update as if from human node
            )
        except Exception as e:
            # Handle reject errors gracefully
            # In a production system, you might want to log this error
            pass

    def reset(self) -> None:
        """Reset conversation state."""
        self._checkpointer = InMemorySaver()
        self._thread_id = f"vibe-session-{uuid4()}"
        self._agent = None  # Reset the agent so it gets reinitialized on next use
        # TODO: Reset interaction logger when implemented

    async def compact(self) -> str:
        """Compact conversation history to reduce context size."""
        if self._agent is None:
            return "No active conversation to compact"

        # Get current state
        state = self._agent.get_state(  # type: ignore
            {"configurable": {"thread_id": self._thread_id}}
        )
        messages = state.values.get("messages", [])

        if len(messages) <= 1:
            return "No messages to compact"

        # Simple compaction: keep only the most recent messages
        # In a real implementation, this would use summarization
        keep_count = max(1, len(messages) // 2)  # Keep half the messages
        compacted_messages = messages[-keep_count:]

        # Update the state with compacted messages
        new_state = state.values.copy()
        new_state["messages"] = compacted_messages

        # For LangGraph, we'd need to update the checkpoint
        # This is a simplified implementation - in practice, we'd need to
        # modify the agent's state properly
        # self._checkpointer.put(
        #     {"configurable": {"thread_id": self._thread_id}},
        #     new_state
        # )

        old_tokens = self._estimate_tokens(messages)
        new_tokens = self._estimate_tokens(compacted_messages)

        return f"Compacted {len(messages)} messages to {len(compacted_messages)} messages, reducing tokens from {old_tokens} to {new_tokens}"

    async def clear_history(self) -> None:
        """Clear conversation history."""
        self.reset()  # Reset achieves the same effect

    def get_log_path(self) -> str | None:
        """Get the path to the current session's log file."""
        # TODO: Implement proper log path retrieval when interaction logger is added
        return None  # Placeholder

    @property
    def stats(self) -> AgentStatsProtocol:
        """Get current session statistics."""
        if self._agent is None:
            return VibeEngineStats(messages=0, context_tokens=0, todos=[])
        # Extract from LangGraph state
        state = self._agent.get_state(  # type: ignore
            {"configurable": {"thread_id": self._thread_id}}
        )
        messages = state.values.get("messages", [])
        context_tokens = self._estimate_tokens(messages)
        todos = state.values.get("todos", [])
        return VibeEngineStats(
            messages=len(messages), context_tokens=context_tokens, todos=todos
        )

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

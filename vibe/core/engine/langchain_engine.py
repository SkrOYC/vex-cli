"""LangChain 1.2.0 powered agent engine for Mistral Vibe."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import Decision, HITLResponse
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_middleware import (
    ContextWarningMiddleware,
    LoggerMiddleware,
    PriceLimitMiddleware,
)
from vibe.core.engine.state import VibeAgentState
from vibe.core.engine.tools import VibeToolAdapter
from vibe.core.engine.tui_events import TUIEventMapper

# Default message used when a tool operation is rejected by the user
_DEFAULT_REJECTION_MESSAGE = "Operation rejected by user"


class VibeEngineStats:
    """Statistics implementation for VibeLangChainEngine that matches AgentStats interface."""

    def __init__(
        self, messages: int = 0, context_tokens: int = 0, todos: list[Any] | None = None
    ) -> None:
        self._messages = messages
        self._todos = todos or []
        self.steps = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.tool_calls_agreed = 0
        self.tool_calls_rejected = 0
        self.tool_calls_failed = 0
        self.tool_calls_succeeded = 0
        self.context_tokens = context_tokens
        self.last_turn_prompt_tokens = 0
        self.last_turn_completion_tokens = 0
        self.last_turn_duration = 0.0
        self.tokens_per_second = 0.0
        self.input_price_per_million = 0.0
        self.output_price_per_million = 0.0

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


class VibeLangChainEngine:
    """LangChain 1.2.0 powered agent engine for Mistral Vibe."""

    def __init__(
        self,
        config: VibeConfig,
    ) -> None:
        self.config = config
        self._agent: CompiledStateGraph | None = None
        self._checkpointer = InMemorySaver()
        self._thread_id = f"vibe-session-{uuid4()}"
        self._stats = VibeEngineStats()
        self._tui_event_mapper: TUIEventMapper | None = None

    def _get_tui_event_mapper(self) -> TUIEventMapper:
        """Lazy initialization of TUI event mapper.

        Returns:
            TUIEventMapper instance for mapping native events to Vibe TUI events
        """
        if self._tui_event_mapper is None:
            self._tui_event_mapper = TUIEventMapper(self.config)
        return self._tui_event_mapper

    def _create_model(self) -> Any:
        """Create LangChain model from Vibe config."""
        from vibe.core.engine.models import create_model_from_config

        return create_model_from_config(self.config)

    def _get_system_prompt(self) -> str:
        """Build system prompt from Vibe config."""
        from vibe.core.system_prompt import get_universal_system_prompt
        from vibe.core.tools.manager import ToolManager

        tool_manager = ToolManager(self.config)
        return get_universal_system_prompt(tool_manager, self.config)

    def _get_pricing_config(self) -> dict[str, tuple[float, float]]:
        """Get pricing configuration from model configs.

        Returns dict mapping model names to (input_rate, output_rate) tuples.
        Rates are per-token (not per-million-tokens).
        """
        pricing = {}
        for model_config in self.config.models:
            pricing[model_config.name] = (
                model_config.input_price / 1_000_000,
                model_config.output_price / 1_000_000,
            )
        return pricing

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
            # Get the active model's name for pricing lookup
            active_model = self.config.get_active_model()
            model_name = active_model.name

            middleware.append(
                PriceLimitMiddleware(
                    max_price=self.config.max_price,
                    model_name=model_name,
                    pricing=self._get_pricing_config(),
                )
            )

        # Human-in-the-loop (native LangChain 1.2.0)
        from vibe.core.engine.permissions import build_interrupt_config

        interrupt_on = build_interrupt_config(self.config)
        if interrupt_on:
            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on=interrupt_on,
                    description_prefix="Tool execution requires approval",
                )
            )

        # Logger middleware for observability
        middleware.append(LoggerMiddleware(enabled=self.config.agent_logging_enabled))

        return middleware

    def initialize(self) -> None:
        """Initialize the LangChain 1.2.0 agent."""
        model = self._create_model()
        tools = VibeToolAdapter.get_all_tools(self.config)

        # LangChain 1.2.0 create_agent() with native middleware stack

        self._agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            middleware=self._build_middleware_stack(),
            state_schema=VibeAgentState,
            checkpointer=self._checkpointer,
            interrupt_before=["tools"],
        )

    async def run(self, user_message: str) -> AsyncGenerator[Any, None]:
        """Run a conversation turn, yielding mapped Vibe TUI events.

        This method streams native LangGraph events through TUIEventMapper
        to convert them to Vibe TUI event types that EventHandler expects.
        """
        if self._agent is None:
            self.initialize()

        assert self._agent is not None

        config: RunnableConfig = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": self.config.max_recursion_depth,
        }

        messages = [("user", user_message)]

        mapper = self._get_tui_event_mapper()

        # Stream native LangGraph events and map to Vibe TUI events
        async for event in self._agent.astream_events(
            {"messages": messages}, config=config, version="v2"
        ):
            # Update stats incrementally from event data
            self._update_stats_from_event(event)

            mapped_event = mapper.map_event(event)
            if mapped_event is not None:
                yield mapped_event

    async def handle_approval(
        self, approved: bool, feedback: str | None = None
    ) -> None:
        """Handle approval decision from TUI using native HITL format.

        Args:
            approved: True to approve, False to reject
            feedback: Optional feedback message for rejections
        """
        if self._agent is None:
            return

        config: RunnableConfig = {"configurable": {"thread_id": self._thread_id}}

        # Build HITLResponse with proper Decision format
        if approved:
            hitl_response = HITLResponse(decisions=[{"type": "approve"}])
        else:
            hitl_response = HITLResponse(
                decisions=[
                    {
                        "type": "reject",
                        "message": feedback or _DEFAULT_REJECTION_MESSAGE,
                    }
                ]
            )

        # Resume with HITLResponse
        await self._agent.ainvoke(Command(resume=hitl_response), config=config)

    async def handle_multi_tool_approval(
        self,
        approvals: list[bool],
        feedbacks: list[str | None],
    ) -> None:
        """Handle approval decisions for multiple interrupted tools.

        Args:
            approvals: List of approval decisions (one per tool)
            feedbacks: List of feedback messages (for rejections)

        Raises:
            ValueError: If lengths don't match
        """
        if self._agent is None:
            return

        if len(approvals) != len(feedbacks):
            raise ValueError(
                f"Length mismatch: {len(approvals)} approvals vs {len(feedbacks)} feedbacks"
            )

        config: RunnableConfig = {"configurable": {"thread_id": self._thread_id}}

        # Build decisions list using list comprehension
        decisions = [
            {"type": "approve"}
            if approved
            else {
                "type": "reject",
                "message": feedback or _DEFAULT_REJECTION_MESSAGE,
            }
            for approved, feedback in zip(approvals, feedbacks, strict=True)
        ]

        hitl_response = HITLResponse(decisions=cast("list[Decision]", decisions))
        await self._agent.ainvoke(Command(resume=hitl_response), config=config)

    async def handle_approve_all(self, tool_count: int) -> None:
        """Approve all interrupted tools.

        Args:
            tool_count: Number of tools to approve
        """
        approvals = [True] * tool_count
        feedbacks: list[str | None] = [None] * tool_count
        await self.handle_multi_tool_approval(approvals, feedbacks)

    async def handle_reject_all(
        self,
        tool_count: int,
        feedback: str | None = None,
    ) -> None:
        """Reject all interrupted tools.

        Args:
            tool_count: Number of tools to reject
            feedback: Rejection feedback (applied to all)
        """
        approvals = [False] * tool_count
        feedbacks: list[str | None] = [feedback] * tool_count
        await self.handle_multi_tool_approval(approvals, feedbacks)

    def reset(self) -> None:
        """Reset conversation state."""
        self._checkpointer = InMemorySaver()
        self._thread_id = f"vibe-session-{uuid4()}"
        self._agent = None

    def compact(self) -> str:
        """Compact conversation history to reduce context size."""
        if self._agent is None:
            return "No active conversation to compact"

        state = self._agent.get_state({"configurable": {"thread_id": self._thread_id}})
        messages = state.values.get("messages", [])

        if len(messages) <= 1:
            return "No messages to compact"

        keep_count = max(1, len(messages) // 2)
        compacted_messages = messages[-keep_count:]

        old_tokens = self._get_actual_token_count(messages)
        new_tokens = self._get_actual_token_count(compacted_messages)

        # Update state with compacted messages
        config: RunnableConfig = {"configurable": {"thread_id": self._thread_id}}
        self._agent.update_state(config, {"messages": compacted_messages})

        return f"Compacted {len(messages)} messages to {len(compacted_messages)} messages, reducing tokens from {old_tokens} to {new_tokens}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.reset()

    def get_current_messages(self) -> list:
        """Get the current conversation messages from the agent state."""
        if self._agent is not None:
            state = self._agent.get_state({
                "configurable": {"thread_id": self._thread_id}
            })
            return state.values.get("messages", [])
        return []

    def get_log_path(self) -> str | None:
        """Get the path to the current session's log file."""
        return None

    @property
    def session_id(self) -> str:
        """Get the session ID/thread ID for this engine."""
        return self._thread_id

    @property
    def stats(self) -> VibeEngineStats:
        """Get current session statistics.

        Note: Stats are updated incrementally during event streaming.
        This property now simply returns the current stats state.
        """
        # Keep message count for backward compatibility with UI
        # (context_tokens is already updated incrementally in run())
        if self._agent is not None:
            state = self._agent.get_state({
                "configurable": {"thread_id": self._thread_id}
            })
            messages = state.values.get("messages", [])
            self._stats._messages = len(messages)

        return self._stats

    def _get_actual_token_count(self, messages: list) -> int:
        """Get actual token count from usage metadata (no estimation!)."""
        total_tokens = 0
        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                usage = msg.usage_metadata
                total_tokens += usage.get("input_tokens", 0)
                total_tokens += usage.get("output_tokens", 0)
        return total_tokens

    def _update_stats_from_event(self, event: dict[str, Any] | Any) -> None:
        """Update stats incrementally from LangGraph event data.

        This method extracts relevant information from LangGraph events
        to incrementally update statistics without needing to recompute
        from the full agent state.

        Args:
            event: A dictionary or StreamEvent representing a LangGraph event from astream_events()
        """
        # Handle both dict and StreamEvent types from astream_events
        event_data = event.__dict__ if not isinstance(event, dict) else event

        event_type = event_data.get("event", "")

        # Handle chat model completion events (token usage)
        if event_type == "on_chat_model_end":
            output = event_data.get("data", {}).get("output", {})
            # The output is typically an AIMessage with usage_metadata
            if hasattr(output, "usage_metadata") and output.usage_metadata:
                usage = output.usage_metadata
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

                # Update session token counts
                self._stats.session_prompt_tokens += input_tokens
                self._stats.session_completion_tokens += output_tokens

                # Update last turn token counts
                self._stats.last_turn_prompt_tokens = input_tokens
                self._stats.last_turn_completion_tokens = output_tokens

                # Update context tokens incrementally
                self._stats.context_tokens += input_tokens + output_tokens

        # Handle tool completion events (step tracking)
        elif event_type == "on_tool_end":
            self._stats.steps += 1

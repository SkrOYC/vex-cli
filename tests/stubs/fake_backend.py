from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterable

from langchain.agents.middleware.human_in_the_loop import HITLRequest, HITLResponse
from langgraph.types import Command

from tests.mock.utils import mock_llm_chunk
from vibe.core.types import LLMChunk, LLMMessage


class FakeBackend:
    """Minimal async backend stub to drive Agent.act without network.

    Provide a finite sequence of LLMResult objects to be returned by
    `complete`. When exhausted, returns an empty assistant message.
    """

    def __init__(
        self,
        results: Iterable[LLMChunk] | None = None,
        *,
        token_counter: Callable[[list[LLMMessage]], int] | None = None,
        exception_to_raise: Exception | None = None,
    ) -> None:
        self._chunks = list(results or [])
        self._requests_messages: list[list[LLMMessage]] = []
        self._requests_extra_headers: list[dict[str, str] | None] = []
        self._count_tokens_calls: list[list[LLMMessage]] = []
        self._token_counter = token_counter or self._default_token_counter
        self._exception_to_raise = exception_to_raise

    @property
    def requests_messages(self) -> list[list[LLMMessage]]:
        return self._requests_messages

    @property
    def requests_extra_headers(self) -> list[dict[str, str] | None]:
        return self._requests_extra_headers

    @staticmethod
    def _default_token_counter(messages: list[LLMMessage]) -> int:
        return 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def complete(
        self,
        *,
        model,
        messages,
        temperature,
        tools,
        tool_choice,
        extra_headers,
        max_tokens,
    ) -> LLMChunk:
        if self._exception_to_raise:
            raise self._exception_to_raise

        self._requests_messages.append(messages)
        self._requests_extra_headers.append(extra_headers)
        if self._chunks:
            chunk = self._chunks.pop(0)
            if not self._chunks:
                chunk = chunk.model_copy(update={"finish_reason": "stop"})
            return chunk
        return mock_llm_chunk(content="", finish_reason="stop")

    async def complete_streaming(
        self,
        *,
        model,
        messages,
        temperature,
        tools,
        tool_choice,
        extra_headers,
        max_tokens,
    ) -> AsyncGenerator[LLMChunk]:
        if self._exception_to_raise:
            raise self._exception_to_raise

        self._requests_messages.append(messages)
        self._requests_extra_headers.append(extra_headers)
        has_final_chunk = False
        while self._chunks:
            chunk = self._chunks.pop(0)
            is_last_provided_chunk = not self._chunks
            if is_last_provided_chunk:
                chunk = chunk.model_copy(update={"finish_reason": "stop"})

            if chunk.finish_reason is not None:
                has_final_chunk = True

            yield chunk
            if has_final_chunk:
                break

        if not has_final_chunk:
            yield mock_llm_chunk(content="", finish_reason="stop")

    async def count_tokens(
        self,
        *,
        model,
        messages,
        temperature=0.0,
        tools,
        tool_choice=None,
        extra_headers,
    ) -> int:
        self._count_tokens_calls.append(list(messages))
        return self._token_counter(messages)


class FakeInterruptedAgent:
    """Mock agent that simulates HITL interrupts.

    Usage:
        agent = FakeInterruptedAgent()
        agent.add_interrupt(
            tool_name="bash",
            args={"command": "ls"},
            description="Execute bash command"
        )

        # Run - will hit interrupt
        # Get interrupt data
        hitl_request = agent.get_interrupt()

        # Resume with decision
        agent.resume(HITLResponse(decisions=[{"type": "approve"}]))
    """

    def __init__(self) -> None:
        self._interrupts: list[HITLRequest] = []
        self._current_interrupt_index = 0
        self._decisions_received: list[HITLResponse] = []
        self._ainvoke_call_count = 0

    def add_interrupt(
        self,
        tool_name: str,
        args: dict,
        description: str,
        allowed_decisions: list[str] | None = None,
    ) -> None:
        """Add a tool interrupt to the sequence."""
        if allowed_decisions is None:
            allowed_decisions = ["approve", "reject"]

        hitl_request = HITLRequest(
            action_requests=[
                {
                    "name": tool_name,
                    "args": args,
                    "description": description,
                }
            ],
            review_configs=[
                {
                    "action_name": tool_name,
                    "allowed_decisions": allowed_decisions,
                }
            ],
        )
        self._interrupts.append(hitl_request)

    def get_interrupt(self) -> HITLRequest | None:
        """Get the next interrupt in the sequence."""
        if self._current_interrupt_index < len(self._interrupts):
            interrupt = self._interrupts[self._current_interrupt_index]
            self._current_interrupt_index += 1
            return interrupt
        return None

    def resume(self, decision: HITLResponse) -> None:
        """Receive a decision for the current interrupt."""
        self._decisions_received.append(decision)

    def get_decisions(self) -> list[HITLResponse]:
        """Get all decisions received so far."""
        return self._decisions_received

    def get_ainvoke_call_count(self) -> int:
        """Get number of times ainvoke was called (for testing)."""
        return self._ainvoke_call_count

    async def ainvoke(self, command: Command, config: dict | None = None) -> None:
        """Mock ainvoke method to accept Command with HITLResponse.

        Args:
            command: Command object containing HITLResponse in resume field
            config: Configuration dict (unused in mock)
        """
        self._ainvoke_call_count += 1

        # Extract HITLResponse from command.resume
        if hasattr(command, "resume") and isinstance(command.resume, dict):
            hitl_response = HITLResponse(**command.resume)
            self._decisions_received.append(hitl_response)


class FakeVibeLangChainEngine:
    """Mock implementation of VibeLangChainEngine for testing.

    This mock provides deterministic test execution without requiring real LLM calls
    or LangGraph state management. It implements the EngineInterface protocol
    and can be configured to yield synthetic events for various test scenarios.

    Usage:
        engine = FakeVibeLangChainEngine(
            config=test_config,
            events_to_yield=[
                AssistantEvent(content="Test response"),
                ToolCallEvent(tool_name="bash", args=...),
                ToolResultEvent(tool_name="bash", result=...),
            ],
        )
        engine.initialize()

        # Stream events
        async for event in engine.run("Hello"):
            process_event(event)
    """

    def __init__(
        self,
        config,
        events_to_yield: list | None = None,
    ) -> None:
        from vibe.core.config import VibeConfig
        from vibe.core.engine.langchain_engine import VibeEngineStats

        self.config: VibeConfig = config
        self._events: list = list(events_to_yield or [])
        self._stats = VibeEngineStats()
        self._session_id = "fake-test-session"
        self._initialized = False
        self._current_run_index = 0

    def initialize(self) -> None:
        """Initialize the fake engine."""
        self._initialized = True

    def _update_stats_from_vibe_event(self, event) -> None:
        """Update stats incrementally from Vibe event types.

        This method mirrors the production VibeLangChainEngine._update_stats_from_event()
        but operates on Vibe event types instead of LangGraph events.

        Args:
            event: A Vibe event (AssistantEvent, ToolCallEvent, ToolResultEvent, etc.)
        """
        from vibe.core.types import AssistantEvent, ToolCallEvent, ToolResultEvent

        if isinstance(event, AssistantEvent):
            self._stats.steps += 1
        elif isinstance(event, ToolCallEvent):
            self._stats.tool_calls_agreed += 1
        elif isinstance(event, ToolResultEvent):
            if event.error:
                self._stats.tool_calls_failed += 1
            else:
                self._stats.tool_calls_succeeded += 1

    async def run(self, user_message: str):
        """Run a conversation turn, yielding synthetic events.

        Args:
            user_message: The user message to process

        Yields:
            BaseEvent instances from the pre-configured event list
        """
        if not self._initialized:
            self.initialize()

        self._current_run_index = 0

        for event in self._events:
            # Update stats incrementally from Vibe event BEFORE yielding
            self._update_stats_from_vibe_event(event)

            yield event
            self._current_run_index += 1

    async def handle_multi_tool_approval(
        self,
        approvals: list[bool],
        feedbacks: list[str | None],
    ) -> None:
        """Receive approval decisions for multiple interrupted tools.

        This method is called by TUI to approve/reject tools.

        Args:
            approvals: List of approval decisions (one per tool)
            feedbacks: List of feedback messages (for rejections)

        Raises:
            ValueError: If lengths don't match
        """
        if len(approvals) != len(feedbacks):
            raise ValueError(
                f"Length mismatch: {len(approvals)} approvals vs {len(feedbacks)} feedbacks"
            )

        for i, (approved, _feedback) in enumerate(
            zip(approvals, feedbacks, strict=True)
        ):
            if approved:
                self._stats.tool_calls_agreed += 1
                if i > 0:
                    self._stats.tool_calls_rejected -= 1  # Already counted
            else:
                self._stats.tool_calls_rejected += 1

    async def clear_history(self) -> None:
        """Clear conversation history."""
        self._events.clear()
        self._stats = self._stats.__class__()
        self._current_run_index = 0

    async def compact(self) -> str:
        """Compact conversation history to reduce context size."""
        if len(self._events) <= 1:
            return "No messages to compact"

        old_count = len(self._events)
        keep_count = max(1, old_count // 2)
        self._events = self._events[-keep_count:]

        return (
            f"Compacted {old_count} events to {keep_count} events, "
            f"reducing from {old_count} to {keep_count}"
        )

    def get_log_path(self) -> str | None:
        """Get the path to the current session's log file."""
        return None

    def get_current_messages(self) -> list:
        """Get current conversation messages from agent state."""
        return []

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    @property
    def stats(self):
        """Get current session statistics."""
        return self._stats

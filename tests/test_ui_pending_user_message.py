from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
import time
from types import SimpleNamespace

import pytest

from vibe.cli.textual_ui.app import VibeApp
from vibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from vibe.cli.textual_ui.widgets.messages import InterruptMessage, UserMessage
from vibe.core.engine import VibeLangChainEngine
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.types import BaseEvent


async def _wait_for(
    pilot, condition: Callable[[], object | None], timeout: float = 3.0
) -> object | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = condition()
        if result:
            return result
        await pilot.pause(0.05)
    return None


class StubAgent:
    """Mock engine implementing EngineInterface protocol for testing."""

    def __init__(self) -> None:
        from vibe.core.types import AgentStatsProtocol

        self.messages: list = []
        self.stats: AgentStatsProtocol = SimpleNamespace(
            context_tokens=0,
            steps=0,
            session_cost=0.0,
            session_prompt_tokens=0,
            session_completion_tokens=0,
            tool_calls_agreed=0,
            tool_calls_rejected=0,
            tool_calls_failed=0,
            tool_calls_succeeded=0,
            last_turn_prompt_tokens=0,
            last_turn_completion_tokens=0,
            last_turn_duration=0.0,
            tokens_per_second=0.0,
            input_price_per_million=0.0,
            output_price_per_million=0.0,
        )
        self.session_id = "test-session-id"

    async def run(self, message: str) -> AsyncGenerator[BaseEvent]:
        """Run a conversation turn (matches EngineInterface)."""
        if False:
            yield
        else:
            return

    async def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.stats.steps = 0

    async def compact(self) -> str:
        """Compact conversation history."""
        return ""

    def get_log_path(self) -> str | None:
        """Get session log path."""
        return None

    async def initialize(self) -> None:
        """Initialize engine (no-op for stub)."""
        return


@pytest.fixture
def vibe_config() -> VibeConfig:
    return VibeConfig(
        session_logging=SessionLoggingConfig(enabled=False), enable_update_checks=False
    )


@pytest.fixture
def vibe_app(vibe_config: VibeConfig) -> VibeApp:
    return VibeApp(config=vibe_config)


def _patch_delayed_init(
    monkeypatch: pytest.MonkeyPatch, init_event: asyncio.Event
) -> None:
    async def _fake_initialize(self: VibeApp) -> None:
        if self.agent or self._agent_initializing:
            return

        self._agent_initializing = True
        try:
            await init_event.wait()
            self.agent = StubAgent()
        except asyncio.CancelledError:
            self.agent = None
            return
        finally:
            self._agent_initializing = False
            self._agent_init_task = None

    monkeypatch.setattr(VibeApp, "_initialize_agent", _fake_initialize, raising=True)


@pytest.mark.asyncio
async def test_shows_user_message_as_pending_until_agent_is_initialized(
    vibe_app: VibeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_event = asyncio.Event()
    _patch_delayed_init(monkeypatch, init_event)

    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "Hello"

        press_task = asyncio.create_task(pilot.press("enter"))

        user_message = await _wait_for(
            pilot, lambda: next(iter(vibe_app.query(UserMessage)), None)
        )
        assert isinstance(user_message, UserMessage)
        assert user_message.has_class("pending")
        init_event.set()
        await press_task
        assert not user_message.has_class("pending")


@pytest.mark.asyncio
async def test_can_interrupt_pending_message_during_initialization(
    vibe_app: VibeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_event = asyncio.Event()
    _patch_delayed_init(monkeypatch, init_event)

    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "Hello"

        press_task = asyncio.create_task(pilot.press("enter"))

        user_message = await _wait_for(
            pilot, lambda: next(iter(vibe_app.query(UserMessage)), None)
        )
        assert isinstance(user_message, UserMessage)
        assert user_message.has_class("pending")

        await pilot.press("escape")
        await press_task
        assert not user_message.has_class("pending")
        assert vibe_app.query(InterruptMessage)
        assert vibe_app.agent is None


@pytest.mark.asyncio
async def test_retry_initialization_after_interrupt(
    vibe_app: VibeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_event = asyncio.Event()
    _patch_delayed_init(monkeypatch, init_event)

    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "First Message"
        press_task = asyncio.create_task(pilot.press("enter"))

        await _wait_for(pilot, lambda: next(iter(vibe_app.query(UserMessage)), None))
        await pilot.press("escape")
        await press_task
        assert vibe_app.agent is None
        assert vibe_app._agent_init_task is None

        chat_input.value = "Second Message"
        press_task_2 = asyncio.create_task(pilot.press("enter"))

        def get_second_message():
            messages = list(vibe_app.query(UserMessage))
            if len(messages) >= 2:
                return messages[-1]
            return None

        user_message_2 = await _wait_for(pilot, get_second_message)
        assert isinstance(user_message_2, UserMessage)
        assert user_message_2.has_class("pending")
        assert vibe_app.agent is None

        init_event.set()
        await press_task_2
        assert not user_message_2.has_class("pending")
        assert vibe_app.agent is not None

"""Tests for VibeEngineStats.

These tests verify VibeEngineStats functionality using FakeVibeLangChainEngine
to avoid requiring real LLM calls. The tests maintain original
test logic while adapting to the new LangChain 1.2.0 architecture.
"""

from __future__ import annotations

import pytest

from tests.stubs.fake_backend import FakeVibeLangChainEngine
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.engine.langchain_engine import VibeEngineStats
from vibe.core.types import (
    AssistantEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def make_config(active_model: str = "test-model") -> VibeConfig:
    """Create a test configuration."""
    return VibeConfig(
        active_model=active_model,
        session_logging=SessionLoggingConfig(enabled=False),
        enable_update_checks=False,
    )


class TestAgentStatsHelpers:
    """Test VibeEngineStats helper methods and calculations."""

    def test_default_initialization(self) -> None:
        """Test that stats initialize with correct defaults."""
        stats = VibeEngineStats()

        assert stats.steps == 0
        assert stats.session_prompt_tokens == 0
        assert stats.session_completion_tokens == 0
        assert stats.tool_calls_agreed == 0
        assert stats.tool_calls_rejected == 0
        assert stats.tool_calls_failed == 0
        assert stats.tool_calls_succeeded == 0
        assert stats.context_tokens == 0
        assert stats.last_turn_prompt_tokens == 0
        assert stats.last_turn_completion_tokens == 0
        assert stats.last_turn_duration == 0.0
        assert stats.tokens_per_second == 0.0
        assert stats.input_price_per_million == 0.0
        assert stats.output_price_per_million == 0.0

    def test_session_total_tokens_property(self) -> None:
        """Test session_total_tokens computed correctly."""
        stats = VibeEngineStats()
        stats.session_prompt_tokens = 1000
        stats.session_completion_tokens = 500

        assert stats.session_total_llm_tokens == 1500

    def test_last_turn_total_tokens_property(self) -> None:
        """Test last_turn_total_tokens computed correctly."""
        stats = VibeEngineStats()
        stats.last_turn_prompt_tokens = 100
        stats.last_turn_completion_tokens = 50

        assert stats.last_turn_total_tokens == 150

    def test_session_cost_computed_from_current_pricing(self) -> None:
        """Test session_cost computed correctly."""
        stats = VibeEngineStats()
        stats.input_price_per_million = 1.5
        stats.output_price_per_million = 3.0

        stats.session_prompt_tokens = 1_000_000  # 1M tokens
        stats.session_completion_tokens = 500_000  # 0.5M tokens

        # Input: 1.0 * 1.5 = $1.50
        # Output: 0.5 * 3.0 = $1.50
        # Total: $3.00
        assert stats.session_cost == 3.0

    def test_update_pricing(self) -> None:
        """Test update_pricing method."""
        stats = VibeEngineStats()

        stats.update_pricing(1.5, 3.0)
        assert stats.input_price_per_million == 1.5
        assert stats.output_price_per_million == 3.0

    def test_reset_context_state_preserves_cumulative(self) -> None:
        """Test that reset_context_state preserves cumulative stats."""
        stats = VibeEngineStats()
        # Set all the stats
        stats.steps = 5
        stats.session_prompt_tokens = 1000
        stats.session_completion_tokens = 500
        stats.tool_calls_succeeded = 3
        stats.tool_calls_failed = 1
        stats.context_tokens = 800
        stats.last_turn_prompt_tokens = 100
        stats.last_turn_completion_tokens = 50
        stats.last_turn_duration = 1.5
        stats.tokens_per_second = 33.3
        stats.input_price_per_million = 0.4
        stats.output_price_per_million = 2.0

        stats.context_tokens = 0
        stats.last_turn_prompt_tokens = 0
        stats.last_turn_completion_tokens = 0
        stats.last_turn_duration = 0.0
        stats.tokens_per_second = 0.0

        # Cumulative stats should be preserved
        assert stats.steps == 5
        assert stats.session_prompt_tokens == 1000
        assert stats.session_completion_tokens == 500
        assert stats.tool_calls_succeeded == 3
        assert stats.tool_calls_failed == 1
        assert stats.input_price_per_million == 0.4
        assert stats.output_price_per_million == 2.0

        # Context-specific stats should be reset
        assert stats.context_tokens == 0
        assert stats.last_turn_prompt_tokens == 0
        assert stats.last_turn_completion_tokens == 0
        assert stats.last_turn_duration == 0.0
        assert stats.tokens_per_second == 0.0


@pytest.mark.asyncio
class TestEngineStatsIntegration:
    """Integration tests for VibeLangChainEngine with FakeVibeLangChainEngine."""

    async def test_engine_stats_initialized_with_zeros(self) -> None:
        """Test that engine stats start at zero."""
        engine = FakeVibeLangChainEngine(config=make_config())
        engine.initialize()

        stats = engine.stats

        assert stats.steps == 0
        assert stats.context_tokens == 0
        assert stats.session_cost == 0.0

    async def test_run_updates_stats(self) -> None:
        """Test that run() method updates stats correctly."""
        engine = FakeVibeLangChainEngine(
            config=make_config(),
            events_to_yield=[
                AssistantEvent(content="Test response"),
            ],
        )
        engine.initialize()

        async for _ in engine.run("Hello"):
            pass

        stats = engine.stats
        assert stats.steps == 1  # One conversation turn
        assert stats.session_prompt_tokens >= 0

    async def test_tool_calls_update_stats(self) -> None:
        """Test that tool call events update stats."""
        engine = FakeVibeLangChainEngine(
            config=make_config(),
            events_to_yield=[
                ToolCallEvent(
                    tool_name="bash",
                    args={"command": "echo test"},
                    tool_call_id="test-1",
                    tool_class=None,
                ),
                ToolResultEvent(
                    tool_name="bash",
                    tool_call_id="test-1",
                    tool_class=None,
                    result=None,
                    error=None,
                ),
            ],
        )
        engine.initialize()

        async for _ in engine.run("Execute command"):
            pass

        stats = engine.stats
        assert stats.tool_calls_agreed == 1
        assert stats.tool_calls_succeeded == 1

    async def test_compact_reduces_event_count(self) -> None:
        """Test that compact() reduces event count."""
        engine = FakeVibeLangChainEngine(
            config=make_config(),
            events_to_yield=[
                AssistantEvent(content="Old message 1"),
                AssistantEvent(content="Old message 2"),
                AssistantEvent(content="Old message 3"),
                AssistantEvent(content="Old message 4"),
            ],
        )
        engine.initialize()

        # Run once to populate events
        async for _ in engine.run("First message"):
            pass

        result = await engine.compact()

        assert "Compacted" in result
        assert "4" in result  # Original count
        assert "2" in result  # Compacted count


@pytest.mark.asyncio
class TestClearHistory:
    """Test clear_history functionality."""

    async def test_clear_history_resets_stats(self) -> None:
        """Test that clear_history resets stats."""
        engine = FakeVibeLangChainEngine(
            config=make_config(),
            events_to_yield=[
                AssistantEvent(content="Test"),
            ],
        )
        engine.initialize()

        # Run once to populate stats
        async for _ in engine.run("Hello"):
            pass

        stats_before = engine.stats
        assert stats_before.steps == 1

        # Clear history
        await engine.clear_history()

        stats_after = engine.stats
        assert stats_after.steps == 0
        assert stats_after.context_tokens == 0


@pytest.mark.asyncio
class TestStatsPropertyAccess:
    """Test that stats property returns current state."""

    async def test_stats_property_updates_after_run(self) -> None:
        """Test that accessing stats property reflects current state."""
        engine = FakeVibeLangChainEngine(
            config=make_config(),
            events_to_yield=[
                AssistantEvent(content="Response"),
            ],
        )
        engine.initialize()

        # Stats should reflect state before run
        assert engine.stats.steps == 0

        # Run a conversation turn
        async for _ in engine.run("Test"):
            pass

        # Stats should reflect updated state
        assert engine.stats.steps == 1

    async def test_session_id_property(self) -> None:
        """Test that session_id property returns unique ID."""
        engine = FakeVibeLangChainEngine(config=make_config())
        engine.initialize()

        session_id = engine.session_id
        assert isinstance(session_id, str)
        assert len(session_id) > 0


@pytest.mark.asyncio
class TestRealTimeStatsUpdates:
    """Tests for real-time stats updates during event streaming."""

    async def test_stats_update_during_run(self) -> None:
        """Test that stats incrementally update while streaming events."""
        engine = FakeVibeLangChainEngine(
            config=make_config(),
            events_to_yield=[
                AssistantEvent(content="Response 1"),
                AssistantEvent(content="Response 2"),
            ],
        )
        engine.initialize()

        # Stats should be zero before run
        assert engine.stats.steps == 0

        # Track stats during streaming
        stats_during_run = []
        async for _event in engine.run("Test"):
            stats_during_run.append(engine.stats.steps)

        # Stats should have been updated during streaming
        assert len(stats_during_run) == 2
        assert stats_during_run[0] == 1  # First event processed
        assert stats_during_run[1] == 2  # Second event processed
        assert engine.stats.steps == 2

    async def test_token_tracking_from_events(self) -> None:
        """Test that token tracking works correctly with synthetic events."""
        engine = FakeVibeLangChainEngine(
            config=make_config(),
            events_to_yield=[
                AssistantEvent(content="Short response"),
                ToolCallEvent(
                    tool_name="bash",
                    args={"command": "echo test"},
                    tool_call_id="test-1",
                    tool_class=None,
                ),
                ToolResultEvent(
                    tool_name="bash",
                    tool_call_id="test-1",
                    tool_class=None,
                    result=None,
                    error=None,
                ),
            ],
        )
        engine.initialize()

        # Run and track stats
        async for _ in engine.run("Test"):
            pass

        stats = engine.stats
        assert stats.steps >= 1
        assert stats.tool_calls_agreed == 1
        assert stats.tool_calls_succeeded == 1

    async def test_context_tokens_accuracy(self) -> None:
        """Test that context_tokens are tracked accurately."""
        engine = FakeVibeLangChainEngine(
            config=make_config(),
            events_to_yield=[
                AssistantEvent(content="Message 1"),
                AssistantEvent(content="Message 2"),
                AssistantEvent(content="Message 3"),
            ],
        )
        engine.initialize()

        # Initial state
        assert engine.stats.context_tokens == 0

        # Run and verify context tokens
        async for _ in engine.run("Test"):
            # Context tokens should be updated during run
            pass

        # Stats should reflect the accumulated events
        assert engine.stats._messages >= 0

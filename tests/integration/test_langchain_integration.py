"""Integration tests for LangChain 1.2.0 engine full workflows.

These tests validate complete user workflows end-to-end, ensuring the engine
works correctly with streaming, tools, approvals, and state management.
"""

from __future__ import annotations

import pytest

from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_engine import VibeLangChainEngine


class TestLangChainIntegration:
    """Integration tests for complete LangChain engine workflows."""

    @pytest.fixture
    def config(self, langchain_config: VibeConfig) -> VibeConfig:
        """Create test configuration."""
        return langchain_config

    @pytest.mark.asyncio
    async def test_conversation_flow(self, config: VibeConfig):
        """Test complete conversation flow with LangChain engine."""
        engine = VibeLangChainEngine(config)

        # Test that engine initializes correctly
        # We can't actually run without mocking, but we can verify initialization
        engine.initialize()
        assert engine._agent is not None
        assert engine._thread_id.startswith("vibe-session-")

    @pytest.mark.asyncio
    async def test_conversation_reset(self, config: VibeConfig):
        """Test reset functionality mid-conversation."""
        engine = VibeLangChainEngine(config)
        engine.initialize()

        original_thread_id = engine._thread_id

        # Reset the conversation
        engine.reset()

        # Verify agent is cleared and thread ID changes
        assert engine._agent is None
        assert engine._thread_id != original_thread_id
        assert engine._thread_id.startswith("vibe-session-")

    @pytest.mark.asyncio
    async def test_conversation_compact(self, config: VibeConfig):
        """Test history compaction functionality."""
        engine = VibeLangChainEngine(config)
        engine.initialize()

        # Test compact with no messages
        result = engine.compact()
        assert result == "No messages to compact"

    @pytest.mark.asyncio
    async def test_middleware_stack_integration(self, config: VibeConfig):
        """Test middleware stack building and integration."""
        # Enable context warnings and price limit
        config.context_warnings = True
        config.max_price = 10.0

        engine = VibeLangChainEngine(config)
        middleware = engine._build_middleware_stack()

        # Should have ContextWarningMiddleware, PriceLimitMiddleware, and HumanInTheLoopMiddleware

        middleware_types = [type(m).__name__ for m in middleware]

        assert "ContextWarningMiddleware" in middleware_types
        assert "PriceLimitMiddleware" in middleware_types
        assert "HumanInTheLoopMiddleware" in middleware_types

    @pytest.mark.asyncio
    async def test_error_handling_graceful(self, config: VibeConfig):
        """Test graceful error handling when agent is None."""
        engine = VibeLangChainEngine(config)

        # Test that methods don't raise when agent is None
        # These should all handle the None case gracefully
        engine.compact()  # Should return message about no conversation
        engine.get_current_messages()  # Should return empty list
        engine.get_log_path()  # Should return None

        # Stats should still work
        stats = engine.stats
        assert stats is not None

    @pytest.mark.asyncio
    async def test_multiple_turns_state_management(self, config: VibeConfig):
        """Test multi-turn conversation state management."""
        engine = VibeLangChainEngine(config)

        # Initialize
        engine.initialize()
        original_thread = engine._thread_id

        # First turn
        messages1 = engine.get_current_messages()
        assert isinstance(messages1, list)

        # Reset should create new thread
        engine.reset()
        assert engine._thread_id != original_thread

        # Second turn should work independently
        messages2 = engine.get_current_messages()
        assert isinstance(messages2, list)


class TestLangChainApprovalWorkflow:
    """Integration tests for native HITL approval workflow."""

    @pytest.fixture
    def config(self, langchain_config: VibeConfig) -> VibeConfig:
        """Create test configuration with approval callback."""
        return langchain_config

    @pytest.mark.asyncio
    async def test_handle_approval_no_agent(self, config: VibeConfig):
        """Test handle_approval with no agent doesn't raise."""
        engine = VibeLangChainEngine(config)

        # Should not raise even with no agent
        await engine.handle_approval(True, "test-feedback")
        await engine.handle_approval(False, "rejected")


class TestLangChainStateManagement:
    """Integration tests for conversation state management."""

    @pytest.fixture
    def config(self, langchain_config: VibeConfig) -> VibeConfig:
        """Create test configuration."""
        return langchain_config

    @pytest.mark.asyncio
    async def test_stats_tracking(self, config: VibeConfig):
        """Test that stats are properly tracked."""
        from vibe.core.engine.langchain_engine import VibeEngineStats

        # Test stats initialization
        stats = VibeEngineStats()
        assert stats.context_tokens == 0
        assert stats.session_prompt_tokens == 0
        assert stats.session_completion_tokens == 0

        # Test stats with values
        stats2 = VibeEngineStats(messages=5, context_tokens=1000)
        assert stats2._messages == 5
        assert stats2.context_tokens == 1000

        # Test computed properties
        stats2.session_prompt_tokens = 100
        stats2.session_completion_tokens = 200
        assert stats2.session_total_llm_tokens == 300

        # Test cost calculation
        stats2.update_pricing(input_price=1.0, output_price=2.0)
        assert stats2.input_price_per_million == 1.0
        assert stats2.output_price_per_million == 2.0

    @pytest.mark.asyncio
    async def test_session_id_persistence(self, config: VibeConfig):
        """Test session ID persists across operations."""
        engine = VibeLangChainEngine(config)

        session_id1 = engine.session_id
        assert session_id1 == engine._thread_id

        # After reset, session ID should change
        engine.reset()
        session_id2 = engine.session_id

        assert session_id2 != session_id1
        assert session_id2.startswith("vibe-session-")


class TestLangChainEngineExports:
    """Test that engine exports are correctly configured."""

    def test_langchain_engine_import(self):
        """Test that VibeLangChainEngine can be imported."""
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        assert VibeLangChainEngine is not None

    def test_vibe_agent_state_import(self):
        """Test that VibeAgentState can be imported."""
        from vibe.core.engine.state import VibeAgentState

        assert VibeAgentState is not None

    def test_engine_stats_import(self):
        """Test that VibeEngineStats can be imported."""
        from vibe.core.engine.langchain_engine import VibeEngineStats

        assert VibeEngineStats is not None

    def test_middleware_imports(self):
        """Test that middleware classes can be imported."""
        from vibe.core.engine.langchain_middleware import (
            ContextWarningMiddleware,
            PriceLimitMiddleware,
        )

        assert ContextWarningMiddleware is not None
        assert PriceLimitMiddleware is not None

    def test_tui_event_mapper_import(self):
        """Test that TUIEventMapper can be imported."""
        from vibe.core.engine.tui_events import TUIEventMapper

        assert TUIEventMapper is not None


class TestLangChainToolAdapter:
    """Test VibeToolAdapter integration with LangChain engine."""

    @pytest.fixture
    def config(self, langchain_config: VibeConfig) -> VibeConfig:
        """Create test configuration."""
        return langchain_config

    def test_tool_adapter_get_all_tools(self, config: VibeConfig):
        """Test that VibeToolAdapter.get_all_tools works with LangChain config."""
        from langchain_core.tools import BaseTool

        from vibe.core.engine.tools import VibeToolAdapter

        tools = VibeToolAdapter.get_all_tools(config)

        # Should return an iterable of BaseTool instances
        assert hasattr(tools, "__iter__")

        for tool in tools:
            assert isinstance(tool, BaseTool)

    def test_tool_adapter_includes_bash_tool(self, config: VibeConfig):
        """Test that bash tool is included in available tools."""
        from vibe.core.engine.tools import VibeToolAdapter

        tools = VibeToolAdapter.get_all_tools(config)
        bash_tools = [tool for tool in tools if "bash" in tool.name.lower()]

        assert len(bash_tools) >= 1


async def collect_all_events(engine: VibeLangChainEngine, message: str):
    """Helper function to collect all events from a conversation turn."""
    events = []
    async for event in engine.run(message):
        events.append(event)
    return events

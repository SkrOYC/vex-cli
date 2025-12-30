"""Feature parity tests for LangChain 1.2.0 migration.

These tests validate that all LangChain 1.2.0 features work correctly
and meet the migration requirements from issues #38, #39, and #40.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
import pytest

from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_engine import VibeEngineStats, VibeLangChainEngine
from vibe.core.engine.langchain_middleware import (
    ContextWarningMiddleware,
    PriceLimitMiddleware,
)
from vibe.core.engine.state import VibeAgentState
from vibe.core.engine.tui_events import TUIEventMapper


class TestNativeEventStreaming:
    """Test native LangGraph event streaming."""

    @pytest.fixture
    def config(self, langchain_config: VibeConfig) -> VibeConfig:
        """Create test configuration."""
        return langchain_config

    @pytest.mark.asyncio
    async def test_run_yields_native_events(self, config: VibeConfig):
        """Test that run() yields native LangGraph events."""
        engine = VibeLangChainEngine(config)
        engine.initialize()

        # Mock the agent to yield test events
        async def mock_astream_events(*args, **kwargs):
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": MagicMock(content="Hello")},
                "run_id": "test-run-id",
            }
            yield {
                "event": "on_tool_start",
                "name": "bash",
                "data": {"input": {"command": "ls"}},
                "run_id": "test-run-id-2",
            }

        engine._agent.astream_events = mock_astream_events  # type: ignore

        # Collect events
        events = []
        async for event in engine.run("test message"):
            events.append(event)

        # Should have received mapped events
        assert len(events) > 0  # Verify events were actually yielded

    @pytest.mark.asyncio
    async def test_auto_initialization_on_run(self, config: VibeConfig):
        """Test that run() auto-initializes if agent is None."""
        engine = VibeLangChainEngine(config)

        # Agent should be None initially
        assert engine._agent is None

        # Mock _create_model to return a properly configured mock
        with patch.object(engine, "_create_model") as mock_create_model:
            mock_model = MagicMock()

            # Create mock agent that can be used in the run() call
            async def mock_astream_events(*args, **kwargs):
                # Return an empty async generator
                if False:
                    yield

            mock_agent = MagicMock()
            mock_agent.astream_events = mock_astream_events
            mock_model.agent = mock_agent

            # Make the model properly awaitable for ainvoke calls
            mock_model.ainvoke = AsyncMock(return_value=MagicMock())

            mock_create_model.return_value = mock_model

            # run() should auto-initialize - verify agent becomes non-None
            # We use a simple test: after calling run(), agent should be initialized
            # Note: This doesn't fully execute the conversation but verifies init path
            try:
                # Try to run - it may fail on agent execution but initialization happens first
                async for _ in engine.run("test message"):
                    pass
            except Exception:
                # Expected - agent execution isn't fully mocked
                pass

            # Verify that _create_model was called (auto-initialization)
            mock_create_model.assert_called_once()

            # Agent should now be set (initialized)
            assert engine._agent is not None


class TestTUIEventMapping:
    """Test TUIEventMapper for native LangGraph events."""

    def test_mapper_handles_chat_model_stream(self):
        """Test that mapper correctly maps on_chat_model_stream events."""
        config = VibeConfig(use_langchain=True)
        mapper = TUIEventMapper(config)

        # Create mock chunk
        mock_chunk = MagicMock()
        mock_chunk.content = "Hello, world!"

        native_event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": mock_chunk},
            "run_id": "test-run-id",
        }

        result = mapper.map_event(native_event)

        # Should return AssistantEvent with content
        from vibe.core.types import AssistantEvent

        assert result is not None
        assert isinstance(result, AssistantEvent)
        assert result.content == "Hello, world!"

    def test_mapper_handles_tool_start(self):
        """Test that mapper correctly maps on_tool_start events."""
        config = VibeConfig(use_langchain=True)
        mapper = TUIEventMapper(config)

        native_event = {
            "event": "on_tool_start",
            "name": "bash",
            "data": {"input": {"command": "ls -la"}},
            "run_id": "test-run-id",
        }

        result = mapper.map_event(native_event)

        from vibe.core.types import ToolCallEvent

        assert result is not None
        assert isinstance(result, ToolCallEvent)
        assert result.tool_name == "bash"

    def test_mapper_handles_tool_end(self):
        """Test that mapper correctly maps on_tool_end events."""
        config = VibeConfig(use_langchain=True)
        mapper = TUIEventMapper(config)

        native_event = {
            "event": "on_tool_end",
            "name": "bash",
            "data": {"output": "file1.txt\nfile2.txt"},
            "run_id": "test-run-id",
        }

        result = mapper.map_event(native_event)

        from vibe.core.types import ToolResultEvent

        assert result is not None
        assert isinstance(result, ToolResultEvent)
        assert result.tool_name == "bash"

    def test_mapper_ignores_unrelated_events(self):
        """Test that mapper ignores unrelated event types."""
        config = VibeConfig(use_langchain=True)
        mapper = TUIEventMapper(config)

        # Test various unrelated event types
        unrelated_events = [
            {"event": "on_chain_start", "name": "my_chain"},
            {"event": "on_agent_start", "name": "agent"},
            {"event": "on_llm_start", "name": "llm"},
        ]

        for event in unrelated_events:
            result = mapper.map_event(event)
            assert result is None, f"Should ignore event type: {event['event']}"


class TestContextWarningAccuracy:
    """Test that context warnings use accurate token counts."""

    def test_warning_with_actual_usage_metadata(self):
        """Test context warning uses actual usage_metadata tokens."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Message with actual token count
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 400,
                "total_tokens": 800,  # 80% of 1000
            },
        )

        state = {"messages": [ai_message]}

        result = middleware.before_model(state, MagicMock())

        assert result is not None
        assert "warning" in result
        assert "80%" in result["warning"]

    def test_no_warning_below_threshold(self):
        """Test no warning is shown below threshold."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Message with tokens below threshold (30% of 1000)
        ai_message = AIMessage(
            content="short response",
            usage_metadata={
                "input_tokens": 150,
                "output_tokens": 150,
                "total_tokens": 300,
            },
        )

        state = {"messages": [ai_message]}

        result = middleware.before_model(state, MagicMock())

        assert result is None  # No warning needed

    def test_no_warning_when_disabled(self):
        """Test no warning when max_context is None."""
        middleware = ContextWarningMiddleware(
            threshold_percent=0.5,
            max_context=None,  # Disabled
        )

        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 5000,
                "output_tokens": 5000,
                "total_tokens": 10000,
            },
        )

        state = {"messages": [ai_message]}

        result = middleware.before_model(state, MagicMock())

        assert result is None  # Warning disabled


class TestPriceLimitEnforcement:
    """Test price limit middleware enforcement."""

    def test_under_limit_allowed(self):
        """Test execution is allowed when under price limit."""
        pricing = {"test-model": (0.000001, 0.000003)}
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing=pricing
        )

        # Small usage - well under $1.00 limit
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )

        state = {"model_name": "test-model", "messages": [ai_message]}

        result = middleware.after_model(state, MagicMock())

        assert result is None  # No error, execution allowed

    def test_over_limit_blocked(self):
        """Test execution is blocked when price limit exceeded."""
        pricing = {"test-model": (0.001, 0.003)}  # $1/$3 per 1k tokens
        middleware = PriceLimitMiddleware(
            max_price=0.001,  # Very low limit
            model_name="test-model",
            pricing=pricing,
        )

        # Large usage that exceeds limit
        ai_message = AIMessage(
            content="expensive response",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 1000,
                "total_tokens": 2000,
            },
        )

        state = {"model_name": "test-model", "messages": [ai_message]}

        with pytest.raises(RuntimeError, match="Price limit exceeded"):
            middleware.after_model(state, MagicMock())


class TestNativeHITLMiddleware:
    """Test native HumanInTheLoopMiddleware integration."""

    def test_hitl_included_in_middleware_stack(self):
        """Test that HITL middleware is included when tools require approval."""
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        from vibe.core.tools.base import BaseToolConfig, ToolPermission

        config = VibeConfig(active_model="test-model", use_langchain=True)

        # Add a tool that requires approval
        bash_config = BaseToolConfig()
        bash_config.permission = ToolPermission.ASK
        config.tools["bash"] = bash_config

        engine = VibeLangChainEngine(config)
        middleware = engine._build_middleware_stack()

        # Should include HumanInTheLoopMiddleware
        hitl_middleware = [
            m for m in middleware if isinstance(m, HumanInTheLoopMiddleware)
        ]

        assert len(hitl_middleware) == 1
        assert hitl_middleware[0].interrupt_on is not None
        assert "bash" in hitl_middleware[0].interrupt_on

    def test_hitl_not_duplicated(self):
        """Test that HITL middleware is not duplicated."""
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        config = VibeConfig(active_model="test-model", use_langchain=True)

        engine = VibeLangChainEngine(config)
        middleware = engine._build_middleware_stack()

        hitl_count = sum(
            1 for m in middleware if isinstance(m, HumanInTheLoopMiddleware)
        )

        assert hitl_count == 1, "HITL middleware should not be duplicated"


class TestVibeAgentStateSchema:
    """Test VibeAgentState schema compliance."""

    def test_state_inherits_from_base(self):
        """Test VibeAgentState inherits from base AgentState."""
        from typing import get_type_hints

        from langchain.agents.middleware.types import AgentState as BaseAgentState

        base_hints = get_type_hints(BaseAgentState)
        vibe_hints = get_type_hints(VibeAgentState)

        # Should have all base fields
        for field in base_hints:
            assert field in vibe_hints, f"Missing base field: {field}"

    def test_state_has_vibe_specific_fields(self):
        """Test VibeAgentState has Vibe-specific fields."""
        annotations = VibeAgentState.__annotations__

        assert "context_tokens" in annotations
        assert "warning" in annotations

        # Verify types
        from typing import get_type_hints

        hints = get_type_hints(VibeAgentState)

        assert hints["context_tokens"] is int
        assert hints["warning"] == str | None

    def test_state_can_be_instantiated(self):
        """Test VibeAgentState can be instantiated."""
        state = VibeAgentState(messages=[], context_tokens=0, warning=None)

        assert state["messages"] == []
        assert state["context_tokens"] == 0
        assert state["warning"] is None


class TestCheckpointerIntegration:
    """Test InMemorySaver checkpointer integration."""

    def test_checkpointer_is_in_memory(self):
        """Test that checkpointer uses InMemorySaver."""
        from langgraph.checkpoint.memory import InMemorySaver

        config = VibeConfig(active_model="test-model", use_langchain=True)

        engine = VibeLangChainEngine(config)

        # Checkpointer should be InMemorySaver
        assert isinstance(engine._checkpointer, InMemorySaver)

    def test_checkpointer_persists_across_operations(self):
        """Test that checkpointer persists state across operations."""
        from langgraph.checkpoint.memory import InMemorySaver

        config = VibeConfig(active_model="test-model", use_langchain=True)

        engine = VibeLangChainEngine(config)

        # Initial checkpointer
        checkpointer1 = engine._checkpointer

        # After reset, checkpointer should be replaced
        engine.reset()
        checkpointer2 = engine._checkpointer

        # Should be different instances
        assert checkpointer1 is not checkpointer2
        # But both should be InMemorySaver
        assert isinstance(checkpointer1, InMemorySaver)
        assert isinstance(checkpointer2, InMemorySaver)


class TestToolAdapterCompatibility:
    """Test VibeToolAdapter compatibility with LangChain."""

    def test_adapter_produces_langchain_tools(self):
        """Test that VibeToolAdapter produces LangChain-compatible tools."""
        from langchain_core.tools import BaseTool

        from vibe.core.config import VibeConfig
        from vibe.core.engine.tools import VibeToolAdapter

        config = VibeConfig(active_model="test-model", use_langchain=True)

        tools = VibeToolAdapter.get_all_tools(config)

        for tool in tools:
            assert isinstance(tool, BaseTool), f"Tool {tool} is not BaseTool"
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "run")

    def test_bash_tool_included(self):
        """Test that bash tool is available."""
        from vibe.core.config import VibeConfig
        from vibe.core.engine.tools import VibeToolAdapter

        config = VibeConfig(active_model="test-model", use_langchain=True)

        tools = VibeToolAdapter.get_all_tools(config)
        bash_tools = [t for t in tools if "bash" in t.name.lower()]

        assert len(bash_tools) >= 1, "Bash tool should be available"


class TestEngineStatistics:
    """Test engine statistics tracking."""

    def test_stats_initialization(self):
        """Test VibeEngineStats initializes correctly."""
        stats = VibeEngineStats()

        assert stats.context_tokens == 0
        assert stats.session_prompt_tokens == 0
        assert stats.session_completion_tokens == 0
        assert stats.steps == 0

    def test_stats_token_properties(self):
        """Test token-related computed properties."""
        stats = VibeEngineStats()

        stats.session_prompt_tokens = 100
        stats.session_completion_tokens = 200

        assert stats.session_total_llm_tokens == 300

        stats.last_turn_prompt_tokens = 50
        stats.last_turn_completion_tokens = 100

        assert stats.last_turn_total_tokens == 150

    def test_stats_cost_calculation(self):
        """Test cost calculation from tokens and pricing."""
        stats = VibeEngineStats()

        stats.session_prompt_tokens = 1_000_000
        stats.session_completion_tokens = 500_000
        stats.update_pricing(input_price=1.0, output_price=2.0)

        # Cost = (1M * $1/M) + (0.5M * $2/M) = $1 + $1 = $2
        assert stats.session_cost == 2.0

    def test_engine_stats_property(self):
        """Test engine.stats property returns VibeEngineStats."""
        from vibe.core.config import VibeConfig

        config = VibeConfig(active_model="test-model", use_langchain=True)

        engine = VibeLangChainEngine(config)

        stats = engine.stats

        assert isinstance(stats, VibeEngineStats)

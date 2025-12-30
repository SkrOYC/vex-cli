"""Performance benchmarks for LangChain 1.2.0 engine.

These benchmarks establish performance baselines to ensure the LangChain 1.2.0
engine meets or exceeds the current DeepAgents engine performance.

See issue #41 for details on performance testing requirements.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vibe.core.config import VibeConfig, SessionLoggingConfig
from vibe.core.engine.langchain_engine import VibeLangChainEngine, VibeEngineStats
from vibe.core.engine.langchain_middleware import (
    ContextWarningMiddleware,
    PriceLimitMiddleware,
)
from vibe.core.engine.state import VibeAgentState
from vibe.core.engine.tui_events import TUIEventMapper
from langchain_core.tools import BaseTool


class TestLangChainEngineBenchmarks:
    """Performance benchmarks for VibeLangChainEngine."""

    @pytest.fixture
    def config(self, langchain_config: VibeConfig) -> VibeConfig:
        """Create test configuration for benchmarks."""
        return langchain_config

    @pytest.mark.benchmark
    def test_engine_initialization_time(self, benchmark, config: VibeConfig):
        """Benchmark engine initialization time.

        This measures how long it takes to create and initialize
        a VibeLangChainEngine instance.
        """

        def initialize_engine():
            engine = VibeLangChainEngine(config)
            engine.initialize()
            return engine

        result = benchmark(initialize_engine)

        # Verify engine was created correctly
        assert result._agent is not None
        assert result._thread_id.startswith("vibe-session-")

    @pytest.mark.benchmark
    def test_middleware_stack_building(self, benchmark, config: VibeConfig):
        """Benchmark middleware stack building time."""

        def build_middleware():
            engine = VibeLangChainEngine(config)
            return engine._build_middleware_stack()

        middleware = benchmark(build_middleware)

        # Verify middleware was built
        assert len(middleware) >= 1  # At least HITL middleware

    @pytest.mark.benchmark
    def test_pricing_config_generation(self, benchmark, config: VibeConfig):
        """Benchmark pricing configuration generation."""

        def get_pricing():
            engine = VibeLangChainEngine(config)
            return engine._get_pricing_config()

        pricing = benchmark(get_pricing)

        # Verify pricing was generated
        assert isinstance(pricing, dict)

    @pytest.mark.benchmark
    def test_stats_creation(self, benchmark):
        """Benchmark VibeEngineStats creation."""

        def create_stats():
            stats = VibeEngineStats(messages=10, context_tokens=1000)
            stats.session_prompt_tokens = 100
            stats.session_completion_tokens = 200
            stats.update_pricing(input_price=1.0, output_price=2.0)
            return stats

        result = benchmark(create_stats)

        # Verify stats were created correctly
        assert result.session_total_llm_tokens == 300
        # Cost = (100/1M * $1) + (200/1M * $2) = $0.0001 + $0.0004 = $0.0005
        assert result.session_cost == 0.0005

    @pytest.mark.benchmark
    def test_reset_operation(self, benchmark, config: VibeConfig):
        """Benchmark engine reset operation."""
        engine = VibeLangChainEngine(config)
        engine.initialize()

        original_thread = engine._thread_id

        def reset_engine():
            engine.reset()
            return engine._thread_id

        new_thread = benchmark(reset_engine)

        # Verify reset worked
        assert new_thread != original_thread
        assert new_thread.startswith("vibe-session-")

    @pytest.mark.benchmark
    def test_get_current_messages_empty(self, benchmark, config: VibeConfig):
        """Benchmark get_current_messages with empty state."""
        engine = VibeLangChainEngine(config)
        engine.initialize()

        def get_messages():
            return engine.get_current_messages()

        messages = benchmark(get_messages)

        # Verify empty list returned
        assert messages == []

    @pytest.mark.benchmark
    def test_session_id_property(self, benchmark, config: VibeConfig):
        """Benchmark session_id property access."""
        engine = VibeLangChainEngine(config)

        def get_session_id():
            return engine.session_id

        session_id = benchmark(get_session_id)

        # Verify session ID is correct
        assert session_id == engine._thread_id
        assert session_id.startswith("vibe-session-")


class TestLangChainMiddlewareBenchmarks:
    """Performance benchmarks for middleware components."""

    @pytest.mark.benchmark
    def test_context_warning_middleware_creation(self, benchmark):
        """Benchmark ContextWarningMiddleware creation."""

        def create_middleware():
            return ContextWarningMiddleware(threshold_percent=0.5, max_context=100000)

        middleware = benchmark(create_middleware)

        # Verify middleware was created
        assert middleware.threshold_percent == 0.5
        assert middleware.max_context == 100000

    @pytest.mark.benchmark
    def test_price_limit_middleware_creation(self, benchmark):
        """Benchmark PriceLimitMiddleware creation."""
        pricing = {"test-model": (0.000001, 0.000003)}

        def create_middleware():
            return PriceLimitMiddleware(
                max_price=10.0, model_name="test-model", pricing=pricing
            )

        middleware = benchmark(create_middleware)

        # Verify middleware was created
        assert middleware.max_price == 10.0
        assert middleware.model_name == "test-model"

    @pytest.mark.benchmark
    def test_context_warning_check_below_threshold(self, benchmark):
        """Benchmark context warning check when below threshold."""
        from langchain_core.messages import AIMessage

        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Message with tokens below threshold
        ai_message = AIMessage(
            content="short",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 100,
                "total_tokens": 200,
            },
        )
        state = {"messages": [ai_message]}

        def check_warning():
            return middleware.before_model(state, MagicMock())

        result = benchmark(check_warning)

        # Should return None (no warning needed)
        assert result is None

    @pytest.mark.benchmark
    def test_price_limit_check_under_limit(self, benchmark):
        """Benchmark price limit check when under limit."""
        from langchain_core.messages import AIMessage

        # Use very low rates to stay under limit
        pricing = {"test-model": (0.00000005, 0.00000005)}
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing=pricing
        )

        # Small usage under limit
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        state = {"model_name": "test-model", "messages": [ai_message]}

        def check_price():
            return middleware.after_model(state, MagicMock())

        result = benchmark(check_price)

        # Should return None (no error)
        assert result is None


class TestLangChainStateBenchmarks:
    """Performance benchmarks for state operations."""

    @pytest.mark.benchmark
    def test_state_creation(self, benchmark):
        """Benchmark VibeAgentState creation."""
        from langchain_core.messages import AIMessage, HumanMessage

        def create_state():
            return VibeAgentState(
                messages=[
                    HumanMessage(content="Hello"),
                    AIMessage(
                        content="Hi there!",
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 20,
                            "total_tokens": 30,
                        },
                    ),
                ],
                context_tokens=30,
                warning=None,
            )

        state = benchmark(create_state)

        # Verify state was created correctly
        assert len(state["messages"]) == 2
        assert state["context_tokens"] == 30

    @pytest.mark.benchmark
    def test_state_token_extraction(self, benchmark):
        """Benchmark token extraction from state."""
        from langchain_core.messages import AIMessage

        state = {
            "messages": [
                AIMessage(
                    content="response",
                    usage_metadata={
                        "input_tokens": 1000,
                        "output_tokens": 500,
                        "total_tokens": 1500,
                    },
                )
            ],
            "context_tokens": 1500,
            "warning": None,
        }

        def extract_tokens():
            total = 0
            for msg in state["messages"]:
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    total += msg.usage_metadata.get("input_tokens", 0)
                    total += msg.usage_metadata.get("output_tokens", 0)
            return total

        tokens = benchmark(extract_tokens)

        assert tokens == 1500


class TestLangChainEventMappingBenchmarks:
    """Performance benchmarks for TUI event mapping."""

    @pytest.fixture
    def config(self) -> VibeConfig:
        """Create test configuration."""
        return VibeConfig(use_langchain=True)

    @pytest.mark.benchmark
    def test_mapper_creation(self, benchmark, config: VibeConfig):
        """Benchmark TUIEventMapper creation."""

        def create_mapper():
            return TUIEventMapper(config)

        mapper = benchmark(create_mapper)

        assert mapper is not None

    @pytest.mark.benchmark
    def test_chat_model_stream_mapping(self, benchmark, config: VibeConfig):
        """Benchmark chat model stream event mapping."""
        from vibe.core.engine.tui_events import TUIEventMapper

        mapper = TUIEventMapper(config)

        mock_chunk = MagicMock()
        mock_chunk.content = "Hello, world!"

        native_event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": mock_chunk},
            "run_id": "test-run-id",
        }

        def map_event():
            return mapper.map_event(native_event)

        result = benchmark(map_event)

        # Should return AssistantEvent
        from vibe.core.types import AssistantEvent

        assert result is not None
        assert isinstance(result, AssistantEvent)

    @pytest.mark.benchmark
    def test_tool_start_mapping(self, benchmark, config: VibeConfig):
        """Benchmark tool start event mapping."""
        from vibe.core.engine.tui_events import TUIEventMapper

        mapper = TUIEventMapper(config)

        native_event = {
            "event": "on_tool_start",
            "name": "bash",
            "data": {"input": {"command": "ls -la"}},
            "run_id": "test-run-id",
        }

        def map_event():
            return mapper.map_event(native_event)

        result = benchmark(map_event)

        # Should return ToolCallEvent
        from vibe.core.types import ToolCallEvent

        assert result is not None
        assert isinstance(result, ToolCallEvent)

    @pytest.mark.benchmark
    def test_tool_end_mapping(self, benchmark, config: VibeConfig):
        """Benchmark tool end event mapping."""
        from vibe.core.engine.tui_events import TUIEventMapper

        mapper = TUIEventMapper(config)

        native_event = {
            "event": "on_tool_end",
            "name": "bash",
            "data": {"output": "file1.txt\nfile2.txt"},
            "run_id": "test-run-id",
        }

        def map_event():
            return mapper.map_event(native_event)

        result = benchmark(map_event)

        # Should return ToolResultEvent
        from vibe.core.types import ToolResultEvent

        assert result is not None
        assert isinstance(result, ToolResultEvent)


class TestLangChainToolAdapterBenchmarks:
    """Performance benchmarks for tool adapter operations."""

    @pytest.fixture
    def config(self, monkeypatch) -> VibeConfig:
        """Create test configuration."""
        monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
        return VibeConfig(active_model="test-model", use_langchain=True)

    @pytest.mark.benchmark
    def test_get_all_tools(self, benchmark, config: VibeConfig):
        """Benchmark VibeToolAdapter.get_all_tools."""
        from vibe.core.engine.tools import VibeToolAdapter

        def get_tools():
            return list(VibeToolAdapter.get_all_tools(config))

        tools = benchmark(get_tools)

        # Should return list of tools
        assert isinstance(tools, list)
        assert len(tools) > 0

    @pytest.mark.benchmark
    def test_bash_tool_creation(self, benchmark, config: VibeConfig):
        """Benchmark bash tool creation."""
        from vibe.core.engine.tools import VibeToolAdapter

        def create_bash_tool():
            return VibeToolAdapter._create_bash_tool(config)

        tool = benchmark(create_bash_tool)

        # Should return a BaseTool
        from langchain_core.tools import BaseTool

        assert isinstance(tool, BaseTool)
        assert "bash" in tool.name

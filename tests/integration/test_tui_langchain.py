"""Integration tests for TUI with VibeLangChainEngine native events.

These tests validate that the TUI correctly consumes native LangGraph events
through TUIEventMapper and handles the complete event flow including
approval dialogs.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_engine import VibeEngineStats, VibeLangChainEngine
from vibe.core.engine.tui_events import TUIEventMapper
from vibe.core.types import AssistantEvent, ToolCallEvent, ToolResultEvent


class TestTUIEventMapper:
    """Test TUIEventMapper functionality for mapping native LangGraph events."""

    @pytest.fixture
    def config(self) -> VibeConfig:
        """Create a test configuration."""
        config = VibeConfig()
        config.use_langchain = True
        return config

    @pytest.fixture
    def mapper(self, config: VibeConfig) -> TUIEventMapper:
        """Create a TUIEventMapper instance."""
        return TUIEventMapper(config)

    def test_mapper_handles_chat_model_stream(self, mapper: TUIEventMapper):
        """Test that TUIEventMapper correctly maps on_chat_model_stream events."""
        # Create a mock chunk with content
        mock_chunk = Mock()
        mock_chunk.content = "Hello, world!"

        native_event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": mock_chunk},
            "run_id": "test-run-id-123",
            "name": "ChatOpenAI",
            "tags": [],
            "metadata": {},
            "parent_ids": [],
        }

        result = mapper.map_event(native_event)

        assert isinstance(result, AssistantEvent)
        assert result.content == "Hello, world!"

    def test_mapper_handles_empty_chat_model_stream(self, mapper: TUIEventMapper):
        """Test that TUIEventMapper handles empty chunks gracefully."""
        mock_chunk = Mock()
        mock_chunk.content = ""

        native_event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": mock_chunk},
            "run_id": "test-run-id",
        }

        result = mapper.map_event(native_event)
        assert result is None

    def test_mapper_handles_tool_start(self, mapper: TUIEventMapper):
        """Test that TUIEventMapper correctly maps on_tool_start events."""
        native_event = {
            "event": "on_tool_start",
            "name": "bash",
            "data": {"input": {"command": "ls -la"}},
            "run_id": "test-run-id-456",
            "tags": [],
            "metadata": {},
            "parent_ids": [],
        }

        result = mapper.map_event(native_event)

        assert isinstance(result, ToolCallEvent)
        assert result.tool_name == "bash"
        assert result.tool_call_id == "test-run-id-456"

    def test_mapper_handles_tool_end(self, mapper: TUIEventMapper):
        """Test that TUIEventMapper correctly maps on_tool_end events."""
        native_event = {
            "event": "on_tool_end",
            "name": "bash",
            "data": {"output": "file1.txt\nfile2.txt\nfile3.txt"},
            "run_id": "test-run-id-789",
            "tags": [],
            "metadata": {},
            "parent_ids": [],
        }

        result = mapper.map_event(native_event)

        assert isinstance(result, ToolResultEvent)
        assert result.tool_name == "bash"
        assert result.tool_call_id == "test-run-id-789"

    def test_mapper_ignores_other_events(self, mapper: TUIEventMapper):
        """Test that TUIEventMapper ignores unrelated event types."""
        native_event = {
            "event": "on_chain_start",
            "name": "my_chain",
            "data": {"input": {"query": "test"}},
            "run_id": "test-run-id",
        }

        result = mapper.map_event(native_event)
        assert result is None

    def test_mapper_handles_dict_conversion(self, mapper: TUIEventMapper):
        """Test that TUIEventMapper handles StreamEvent objects with __dict__."""
        # Create a mock object that mimics StreamEvent
        mock_chunk = Mock()
        mock_chunk.content = "Test response"

        mock_event = Mock()
        mock_event.__dict__ = {
            "event": "on_chat_model_stream",
            "data": {"chunk": mock_chunk},
            "run_id": "test-run-id",
        }

        result = mapper.map_event(mock_event)

        assert isinstance(result, AssistantEvent)
        assert result.content == "Test response"

    def test_mapper_handles_unknown_tool(self, mapper: TUIEventMapper):
        """Test that TUIEventMapper handles unknown tools with fallback."""
        native_event = {
            "event": "on_tool_start",
            "name": "unknown_tool_xyz",
            "data": {"input": {"param1": "value1", "param2": "value2"}},
            "run_id": "test-run-id",
        }

        result = mapper.map_event(native_event)

        assert isinstance(result, ToolCallEvent)
        assert result.tool_name == "unknown_tool_xyz"
        assert result.tool_class is None  # Unknown tool has no class
        # Check that args were stored correctly
        # For GenericArgs, the args dict is stored in the 'args' field
        args_field = getattr(result.args, "args", None)
        assert args_field == {"param1": "value1", "param2": "value2"}


class TestVibeLangChainEngineWithMapper:
    """Test VibeLangChainEngine integration with TUIEventMapper."""

    @pytest.fixture
    def config(self) -> VibeConfig:
        """Create a test configuration."""
        config = VibeConfig()
        config.use_langchain = True
        return config

    def test_engine_initialization(self, config: VibeConfig):
        """Test that engine initializes correctly."""
        engine = VibeLangChainEngine(config)

        assert engine._agent is None
        assert engine.config == config
        assert engine._tui_event_mapper is None  # Lazy initialization
        assert isinstance(engine._stats, VibeEngineStats)

    def test_engine_lazy_mapper_initialization(self, config: VibeConfig):
        """Test that TUIEventMapper is lazily initialized."""
        engine = VibeLangChainEngine(config)

        # Mapper should be None initially
        assert engine._tui_event_mapper is None

        # Getting mapper should create it
        mapper = engine._get_tui_event_mapper()
        assert mapper is not None
        assert isinstance(mapper, TUIEventMapper)

        # Second call should return same instance
        mapper2 = engine._get_tui_event_mapper()
        assert mapper is mapper2

        # Mapper should now be cached
        assert engine._tui_event_mapper is mapper


class TestEngineExports:
    """Test that engine exports are correctly configured."""

    def test_tui_event_mapper_is_exported(self):
        """Test that TUIEventMapper is exported from engine module."""
        from vibe.core.engine import TUIEventMapper

        assert TUIEventMapper is not None

    def test_langchain_engine_import(self):
        """Test that VibeLangChainEngine can be imported."""
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        assert VibeLangChainEngine is not None

    def test_engine_init_includes_tui_event_mapper(self):
        """Test that engine __init__ includes TUIEventMapper import."""
        import vibe.core.engine.langchain_engine as langchain_engine

        # TUIEventMapper should be imported
        assert hasattr(langchain_engine, "TUIEventMapper")


class TestDeprecationWarnings:
    """Test that deprecated classes emit warnings."""

    def test_event_translator_deprecation_warning(self):
        """Test that EventTranslator emits deprecation warning."""
        import warnings

        from vibe.core.config import VibeConfig
        from vibe.core.engine.adapters import EventTranslator

        config = VibeConfig()
        config.use_langchain = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EventTranslator(config)

            assert len(w) >= 1
            assert any(
                "EventTranslator is deprecated" in str(warning.message) for warning in w
            )

    def test_approval_bridge_deprecation_warning(self):
        """Test that ApprovalBridge emits deprecation warning."""
        import warnings

        from vibe.core.config import VibeConfig
        from vibe.core.engine.adapters import ApprovalBridge

        config = VibeConfig()
        config.use_langchain = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ApprovalBridge(config)

            assert len(w) >= 1
            assert any(
                "ApprovalBridge is deprecated" in str(warning.message) for warning in w
            )

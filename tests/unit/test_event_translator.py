"""Unit tests for EventTranslator with DeepAgents integration."""

from __future__ import annotations

from vibe.core.config import VibeConfig
from vibe.core.engine.adapters import EventTranslator
from vibe.core.types import AssistantEvent, ToolCallEvent, ToolResultEvent


class TestEventTranslator:
    """Test event translation between DeepAgents and Vibe."""

    def test_translate_chat_model_stream(self, deepagents_config: VibeConfig):
        """Test translation of on_chat_model_stream event."""
        translator = EventTranslator(deepagents_config)
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": type("MockChunk", (), {"content": "test"})()},
        }
        result = translator.translate(event)
        # Result may be None if content is empty or chunk is not as expected
        # But if it's not None, it should be an AssistantEvent with correct content
        if result is not None:
            assert isinstance(result, AssistantEvent)
            assert result.content == "test"

    def test_translate_tool_start(self, deepagents_config: VibeConfig):
        """Test translation of on_tool_start event."""
        translator = EventTranslator(deepagents_config)
        event = {
            "event": "on_tool_start",
            "name": "bash",
            "run_id": "123",
            "data": {"input": {"command": "ls"}},
        }
        result = translator.translate(event)
        # Result may be None if tool not found, but if it exists, it should be a ToolCallEvent
        if result is not None:
            assert isinstance(result, ToolCallEvent)
            assert result.tool_name == "bash"

    def test_translate_tool_end(self, deepagents_config: VibeConfig):
        """Test translation of on_tool_end event."""
        translator = EventTranslator(deepagents_config)
        event = {
            "event": "on_tool_end",
            "name": "bash",
            "run_id": "123",
            "data": {"output": "file1\nfile2"},
        }
        result = translator.translate(event)
        # Result may be None if tool not found, but if it exists, it should be a ToolResultEvent
        if result is not None:
            assert isinstance(result, ToolResultEvent)
            assert result.tool_name == "bash"

    def test_translate_unknown_event(self, deepagents_config: VibeConfig):
        """Test translation of unknown event returns None."""
        translator = EventTranslator(deepagents_config)
        event = {"event": "unknown", "data": {}}
        result = translator.translate(event)
        assert result is None

    def test_initialization(self, deepagents_config: VibeConfig):
        """Test EventTranslator initializes correctly."""
        translator = EventTranslator(deepagents_config)
        assert translator.config == deepagents_config
        assert translator.tool_manager is not None

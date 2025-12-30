"""Tests for EventTranslator."""

from __future__ import annotations

from vibe.core.config import VibeConfig
from vibe.core.engine.adapters import EventTranslator


class TestEventTranslator:
    def test_translate_chat_model_stream(self):
        """Test translation of on_chat_model_stream event."""
        config = VibeConfig()
        translator = EventTranslator(config)
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": type("MockChunk", (), {"content": "test"})()},
        }
        result = translator.translate(event)
        assert result is not None
        assert result.content == "test"

    def test_translate_tool_start(self):
        """Test translation of on_tool_start event."""
        config = VibeConfig()
        translator = EventTranslator(config)
        event = {
            "event": "on_tool_start",
            "name": "bash",
            "run_id": "123",
            "data": {"input": {"command": "ls"}},
        }
        result = translator.translate(event)
        assert result is not None
        assert result.tool_name == "bash"
        assert result.tool_call_id == "123"
        # Note: tool_class may be None if tool not found, args should contain the input

    def test_translate_tool_end(self):
        """Test translation of on_tool_end event."""
        config = VibeConfig()
        translator = EventTranslator(config)
        event = {
            "event": "on_tool_end",
            "name": "bash",
            "run_id": "123",
            "data": {"output": "file1\nfile2"},
        }
        result = translator.translate(event)
        assert result is not None
        assert result.tool_name == "bash"
        assert result.tool_call_id == "123"
        # Note: tool_class may be None if tool not found, result should contain the output

    def test_translate_unknown_event(self):
        """Test translation of unknown event returns None."""
        config = VibeConfig()
        translator = EventTranslator(config)
        event = {"event": "unknown", "data": {}}
        result = translator.translate(event)
        assert result is None

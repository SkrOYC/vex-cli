"""Tests for EventTranslator."""

import pytest
from vibe.core.engine.adapters import EventTranslator


class TestEventTranslator:
    def test_translate_chat_model_stream(self):
        """Test translation of on_chat_model_stream event."""
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": type("MockChunk", (), {"content": "test"})()}
        }
        result = EventTranslator.translate(event)
        assert result is not None
        assert result.content == "test"

    def test_translate_tool_start(self):
        """Test translation of on_tool_start event."""
        event = {
            "event": "on_tool_start",
            "name": "bash",
            "run_id": "123",
            "data": {"input": {"command": "ls"}}
        }
        result = EventTranslator.translate(event)
        assert result is not None
        assert result.tool_name == "bash"
        assert result.tool_call_id == "123"

    def test_translate_tool_end(self):
        """Test translation of on_tool_end event."""
        event = {
            "event": "on_tool_end",
            "name": "bash",
            "run_id": "123",
            "data": {"output": "file1\nfile2"}
        }
        result = EventTranslator.translate(event)
        assert result is not None
        assert result.tool_name == "bash"
        assert result.tool_call_id == "123"

    def test_translate_unknown_event(self):
        """Test translation of unknown event returns None."""
        event = {"event": "unknown", "data": {}}
        result = EventTranslator.translate(event)
        assert result is None
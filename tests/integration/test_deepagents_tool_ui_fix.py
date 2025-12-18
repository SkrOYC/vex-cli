"""Integration test specifically for ToolUIDataAdapter DeepAgents fix."""

from __future__ import annotations

import pytest

from vibe.core.engine.adapters import EventTranslator
from vibe.core.tools.ui import ToolUIDataAdapter
from vibe.core.types import ToolResultEvent


def test_event_translator_creates_tool_result_event_with_none_tool_class():
    """Test that EventTranslator creates ToolResultEvent with tool_class=None for DeepAgents tools."""
    # This simulates the scenario described in the issue where DeepAgents middleware
    # provides tools that are not registered in the legacy ToolManager
    
    # Create a ToolResultEvent with tool_class=None as DeepAgents would do
    event = ToolResultEvent(
        tool_name="deepagents_tool",
        tool_class=None,  # This is the key part of the test
        tool_call_id="test_id",
        result=None
    )
    
    # This should not crash the ToolUIDataAdapter
    adapter = ToolUIDataAdapter(None)
    display = adapter.get_result_display(event)
    
    # Verify the adapter handles None tool_class gracefully
    assert display is not None
    assert display.success is True


def test_tool_ui_adapter_with_none_tool_class_no_crash():
    """Direct test that ToolUIDataAdapter with None tool_class doesn't crash."""
    # This directly tests the fix from the issue
    adapter = ToolUIDataAdapter(None)
    
    # The critical test: this should not raise TypeError: issubclass() arg 1 must be a class
    assert adapter.ui_data_class is None
    assert adapter.tool_class is None
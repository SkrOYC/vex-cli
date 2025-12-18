"""Test ToolUIDataAdapter with various tool_class scenarios."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from vibe.core.tools.ui import ToolUIDataAdapter, ToolCallDisplay, ToolResultDisplay
from vibe.core.types import ToolCallEvent, ToolResultEvent
from vibe.core.tools.base import BaseTool


@pytest.fixture
def none_tool_adapter():
    """Provides a ToolUIDataAdapter initialized with a None tool_class."""
    return ToolUIDataAdapter(None)


class MockToolArgs(BaseModel):
    """Mock tool arguments for testing."""
    message: str
    count: int = 1


class MockToolResult(BaseModel):
    """Mock tool result for testing."""
    status: str
    data: dict[str, Any]


class NonUIDataTool(BaseTool):
    """Mock tool class that doesn't implement ToolUIData protocol."""
    pass


def test_tool_ui_adapter_with_none_tool_class(none_tool_adapter):
    """Test ToolUIDataAdapter handles None tool_class gracefully."""
    assert none_tool_adapter.tool_class is None
    assert none_tool_adapter.ui_data_class is None


def test_tool_ui_adapter_with_non_ui_data_tool():
    """Test ToolUIDataAdapter with tool class that doesn't implement ToolUIData."""
    adapter = ToolUIDataAdapter(NonUIDataTool)
    
    assert adapter.tool_class is NonUIDataTool
    assert adapter.ui_data_class is None


def test_get_call_display_with_none_tool_class(none_tool_adapter):
    """Test get_call_display works correctly when tool_class is None."""
    
    # Create a mock ToolCallEvent
    mock_args = MockToolArgs(message="test", count=2)
    event = ToolCallEvent(
        tool_name="test_tool",
        tool_class=None,
        args=mock_args,
        tool_call_id="test_id"
    )
    
    display = none_tool_adapter.get_call_display(event)
    
    assert isinstance(display, ToolCallDisplay)
    assert "test_tool(message='test', count=2)" in display.summary
    assert display.details == {"message": "test", "count": 2}


def test_get_result_display_with_none_tool_class_error(none_tool_adapter):
    """Test get_result_display handles error case when tool_class is None."""
    
    event = ToolResultEvent(
        tool_name="test_tool",
        tool_class=None,
        tool_call_id="test_id",
        error="Test error message"
    )
    
    display = none_tool_adapter.get_result_display(event)
    
    assert isinstance(display, ToolResultDisplay)
    assert display.success is False
    assert display.message == "Test error message"


def test_get_result_display_with_none_tool_class_skipped(none_tool_adapter):
    """Test get_result_display handles skipped case when tool_class is None."""
    
    event = ToolResultEvent(
        tool_name="test_tool",
        tool_class=None,
        tool_call_id="test_id",
        skipped=True,
        skip_reason="Test skip reason"
    )
    
    display = none_tool_adapter.get_result_display(event)
    
    assert isinstance(display, ToolResultDisplay)
    assert display.success is False
    assert display.message == "Test skip reason"


def test_get_result_display_with_none_tool_class_success(none_tool_adapter):
    """Test get_result_display handles success case when tool_class is None."""
    
    mock_result = MockToolResult(status="success", data={"key": "value"})
    event = ToolResultEvent(
        tool_name="test_tool",
        tool_class=None,
        tool_call_id="test_id",
        result=mock_result
    )
    
    display = none_tool_adapter.get_result_display(event)
    
    assert isinstance(display, ToolResultDisplay)
    assert display.success is True
    assert display.message == "Success"
    assert display.details == {"status": "success", "data": {"key": "value"}}
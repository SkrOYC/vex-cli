"""Test tool result display works correctly with various tools."""

from __future__ import annotations

from typing import Any
import pytest
from pydantic import BaseModel

from vibe.core.tools.ui import ToolUIDataAdapter, ToolCallDisplay, ToolResultDisplay
from vibe.core.types import ToolCallEvent, ToolResultEvent


@pytest.fixture
def none_tool_adapter():
    """Provides a ToolUIDataAdapter initialized with a None tool_class."""
    return ToolUIDataAdapter(None)


class MockToolArgs(BaseModel):
    """Mock tool arguments."""
    message: str
    count: int = 1


class MockToolResult(BaseModel):
    """Mock tool result."""
    status: str
    data: dict[str, Any]


def test_tool_result_display_with_bash_tool(none_tool_adapter):
    """Test tool result display with bash tool simulation."""
    
    # Simulate bash tool result with BaseModel result
    class BashResult(BaseModel):
        output: str
        exit_code: int
    
    bash_result = BashResult(output="Hello World", exit_code=0)
    event = ToolResultEvent(
        tool_name="bash",
        tool_class=None,
        tool_call_id="bash_123",
        result=bash_result
    )
    
    display = none_tool_adapter.get_result_display(event)
    
    assert display.success is True
    assert display.message == "Success"
    assert display.details == {"output": "Hello World", "exit_code": 0}


def test_tool_result_display_with_read_file_tool(none_tool_adapter):
    """Test tool result display with read_file tool simulation."""
    
    # Simulate read_file result
    mock_result = MockToolResult(
        status="success",
        data={"content": "file content", "path": "/path/to/file.txt"}
    )
    event = ToolResultEvent(
        tool_name="read_file",
        tool_class=None,
        tool_call_id="read_123",
        result=mock_result
    )
    
    display = none_tool_adapter.get_result_display(event)
    
    assert display.success is True
    assert display.message == "Success"
    assert display.details == {
        "status": "success",
        "data": {"content": "file content", "path": "/path/to/file.txt"}
    }


def test_tool_result_display_with_error_result(none_tool_adapter):
    """Test tool result display with error."""
    
    event = ToolResultEvent(
        tool_name="bash",
        tool_class=None,
        tool_call_id="bash_123",
        error="Command failed with exit code 1"
    )
    
    display = none_tool_adapter.get_result_display(event)
    
    assert display.success is False
    assert display.message == "Command failed with exit code 1"
    assert display.details == {}


def test_tool_result_display_with_skipped_result(none_tool_adapter):
    """Test tool result display with skipped tool."""
    
    event = ToolResultEvent(
        tool_name="write_file",
        tool_class=None,
        tool_call_id="write_123",
        skipped=True,
        skip_reason="User denied permission"
    )
    
    display = none_tool_adapter.get_result_display(event)
    
    assert display.success is False
    assert display.message == "User denied permission"
    assert display.details == {}


def test_tool_call_display_with_various_arg_types(none_tool_adapter):
    """Test tool call display with different argument types."""
    
    # Test with complex arguments
    complex_args = MockToolArgs(
        message="test message",
        count=42
    )
    event = ToolCallEvent(
        tool_name="complex_tool",
        tool_class=None,
        args=complex_args,
        tool_call_id="complex_123"
    )
    
    display = none_tool_adapter.get_call_display(event)
    
    assert display.summary == "complex_tool(message='test message', count=42)"
    assert display.details == {"message": "test message", "count": 42}


def test_tool_call_display_with_no_args(none_tool_adapter):
    """Test tool call display with no arguments."""
    
    # Create minimal args model
    class NoArgs(BaseModel):
        pass
    
    event = ToolCallEvent(
        tool_name="no_args_tool",
        tool_class=None,
        args=NoArgs(),
        tool_call_id="noargs_123"
    )
    
    display = none_tool_adapter.get_call_display(event)
    
    assert display.summary == "no_args_tool()"
    assert display.details == {}


def test_tool_result_display_with_none_result(none_tool_adapter):
    """Test tool result display with None result (successful but no output)."""
    
    event = ToolResultEvent(
        tool_name="silent_tool",
        tool_class=None,
        tool_call_id="silent_123",
        result=None
    )
    
    display = none_tool_adapter.get_result_display(event)
    
    assert display.success is True
    assert display.message == "Success"
    assert display.details == {}
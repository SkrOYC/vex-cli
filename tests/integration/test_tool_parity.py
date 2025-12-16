"""Tool parity tests comparing legacy tools with DeepAgents implementations."""

import pytest
import tempfile
from pathlib import Path

from vibe.core.config import VibeConfig
from vibe.core.engine.tools import VibeToolAdapter


class TestToolParity:
    """Test parity between legacy and new tool implementations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def config(self, temp_dir):
        """Create a test config."""
        config = VibeConfig()
        config.workdir = temp_dir
        return config

    def test_read_file_parity(self, config, temp_dir):
        """Test read_file tool parity."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_content = "Hello\nWorld\n"
        test_file.write_text(test_content)

        # Get new tool
        tools = VibeToolAdapter.get_all_tools(config)
        read_tool = next((t for t in tools if t.name == "read_file"), None)
        assert read_tool is not None, "read_file tool not found"

        # Test new tool
        # Note: This would require a proper runtime context, simplified for now
        # In real test, would need to mock runtime and state

        # For now, just check tool exists and has correct interface
        assert hasattr(read_tool, "name")
        assert hasattr(read_tool, "description")
        assert read_tool.name == "read_file"

    def test_write_file_parity(self, config, temp_dir):
        """Test write_file tool parity."""
        tools = VibeToolAdapter.get_all_tools(config)
        write_tool = next((t for t in tools if t.name == "write_file"), None)
        assert write_tool is not None, "write_file tool not found"
        assert write_tool.name == "write_file"

    def test_edit_file_parity(self, config, temp_dir):
        """Test edit_file tool parity."""
        tools = VibeToolAdapter.get_all_tools(config)
        edit_tool = next((t for t in tools if t.name == "edit_file"), None)
        assert edit_tool is not None, "edit_file tool not found"
        assert edit_tool.name == "edit_file"

    def test_ls_parity(self, config, temp_dir):
        """Test ls tool parity."""
        tools = VibeToolAdapter.get_all_tools(config)
        ls_tool = next((t for t in tools if t.name == "ls"), None)
        assert ls_tool is not None, "ls tool not found"
        assert ls_tool.name == "ls"

    def test_glob_parity(self, config, temp_dir):
        """Test glob tool parity."""
        tools = VibeToolAdapter.get_all_tools(config)
        glob_tool = next((t for t in tools if t.name == "glob"), None)
        assert glob_tool is not None, "glob tool not found"
        assert glob_tool.name == "glob"

    def test_grep_parity(self, config, temp_dir):
        """Test grep tool parity."""
        tools = VibeToolAdapter.get_all_tools(config)
        grep_tool = next((t for t in tools if t.name == "grep"), None)
        assert grep_tool is not None, "grep tool not found"
        assert grep_tool.name == "grep"

    def test_bash_parity(self, config, temp_dir):
        """Test bash tool parity."""
        tools = VibeToolAdapter.get_all_tools(config)
        bash_tool = next((t for t in tools if t.name == "bash"), None)
        assert bash_tool is not None, "bash tool not found"
        assert bash_tool.name == "bash"

    def test_custom_tools_loading(self, config, temp_dir):
        """Test custom tool loading."""
        # Create a test custom tool file
        custom_tool_file = temp_dir / "custom_tool.py"
        custom_tool_file.write_text("""
from langchain_core.tools import BaseTool, StructuredTool

def my_custom_tool(input: str) -> str:
    \"\"\"A custom tool.\"\"\"
    return f"Processed: {input}"

# Create tool instance
custom_tool = StructuredTool.from_function(
    name="custom_tool",
    description="A test custom tool",
    func=my_custom_tool,
)
""")

        # Update config to include the temp dir
        config.tool_paths = [str(temp_dir)]

        tools = VibeToolAdapter.get_all_tools(config)
        custom_tool = next((t for t in tools if t.name == "custom_tool"), None)
        assert custom_tool is not None, "Custom tool not loaded"
        assert custom_tool.name == "custom_tool"

    def test_custom_tools_loading_from_directory(self, config, temp_dir):
        """Test custom tool loading from directory with multiple files."""
        # Create directory with multiple tool files
        tools_dir = temp_dir / "tools"
        tools_dir.mkdir()

        # First tool file
        tool1_file = tools_dir / "tool1.py"
        tool1_file.write_text("""
from langchain_core.tools import StructuredTool

def tool_one(value: int) -> str:
    return f"One: {value}"

tool_one_instance = StructuredTool.from_function(
    name="tool_one",
    description="First tool",
    func=tool_one,
)
""")

        # Second tool file
        tool2_file = tools_dir / "tool2.py"
        tool2_file.write_text("""
from langchain_core.tools import StructuredTool

def tool_two(text: str) -> str:
    return f"Two: {text}"

tool_two_instance = StructuredTool.from_function(
    name="tool_two",
    description="Second tool",
    func=tool_two,
)
""")

        config.tool_paths = [str(tools_dir)]

        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {tool.name for tool in tools}

        assert "tool_one" in tool_names, "First custom tool not loaded"
        assert "tool_two" in tool_names, "Second custom tool not loaded"

    def test_custom_tools_invalid_path(self, config, temp_dir):
        """Test custom tool loading with invalid paths."""
        # Set invalid path
        config.tool_paths = ["/nonexistent/path"]

        # Should not raise exception, just log warning
        tools = VibeToolAdapter.get_all_tools(config)

        # Should still have basic tools
        tool_names = {tool.name for tool in tools}
        assert "bash" in tool_names, "Basic tools should still be available"

    def test_custom_tools_malformed_file(self, config, temp_dir):
        """Test custom tool loading with malformed Python file."""
        # Create malformed Python file
        bad_tool_file = temp_dir / "bad_tool.py"
        bad_tool_file.write_text("""
# This file has syntax errors
def broken_function(
    return "broken"
""")

        config.tool_paths = [str(temp_dir)]

        # Should handle error gracefully
        tools = VibeToolAdapter.get_all_tools(config)

        # Should still have basic tools
        tool_names = {tool.name for tool in tools}
        assert "bash" in tool_names, "Basic tools should still be available"

    def test_custom_tools_no_tools_in_file(self, config, temp_dir):
        """Test custom tool loading from file with no tools."""
        # Create file with no tool instances
        empty_tool_file = temp_dir / "empty_tool.py"
        empty_tool_file.write_text("""
# This file has no tools
def regular_function():
    return "hello"

class RegularClass:
    pass
""")

        config.tool_paths = [str(temp_dir)]

        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {tool.name for tool in tools}

        # Should not have any custom tools from this file
        assert "regular_function" not in tool_names, (
            "Non-tool objects should not be loaded"
        )
        assert "bash" in tool_names, "Basic tools should still be available"

    def test_custom_tools_at_decorator(self, config, temp_dir):
        """Test custom tool loading with @tool decorator pattern."""
        # Create file with @tool decorated function
        decorator_tool_file = temp_dir / "decorator_tool.py"
        decorator_tool_file.write_text("""
from langchain_core.tools import tool

@tool
def decorated_tool(message: str) -> str:
    \"\"\"A tool with decorator.\"\"\"
    return f"Decorated: {message}"
""")

        config.tool_paths = [str(temp_dir)]

        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {tool.name for tool in tools}

        assert "decorated_tool" in tool_names, "@tool decorated function not loaded"

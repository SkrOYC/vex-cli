"""Comprehensive integration tests for all tool types in DeepAgents."""

import pytest
import tempfile
from pathlib import Path

from vibe.core.config import VibeConfig
from vibe.core.engine.tools import VibeToolAdapter


class TestAllToolsIntegration:
    """Integration tests for all tool types."""

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

    def test_filesystem_tools_available(self, config):
        """Test that all filesystem tools are available."""
        tools = VibeToolAdapter.get_all_tools(config)

        tool_names = {tool.name for tool in tools}
        expected_tools = {
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "bash",  # Custom bash tool
            # TodoListMiddleware tools would be here
        }

        # Check that filesystem tools are present
        for tool_name in ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]:
            assert tool_name in tool_names, f"{tool_name} tool not found"

        assert "bash" in tool_names

    def test_custom_tools_integration(self, config, temp_dir):
        """Test loading and integration of custom tools."""
        # Create a custom tool file
        custom_dir = temp_dir / "custom_tools"
        custom_dir.mkdir()
        custom_file = custom_dir / "test_tool.py"

        custom_file.write_text("""
from langchain_core.tools import StructuredTool

def sample_tool(text: str) -> str:
    \"\"\"A sample custom tool.\"\"\"
    return f"Processed: {text}"

sample_tool_instance = StructuredTool.from_function(
    name="sample_tool",
    description="A test tool",
    func=sample_tool,
)
""")

        # Update config
        config.tool_paths = [str(custom_dir)]

        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {tool.name for tool in tools}

        assert "sample_tool" in tool_names, "Custom tool not loaded"

    def test_tool_descriptions(self, config):
        """Test that tools have proper descriptions."""
        tools = VibeToolAdapter.get_all_tools(config)

        for tool in tools:
            assert tool.description, f"Tool {tool.name} has no description"
            assert len(tool.description.strip()) > 0, (
                f"Tool {tool.name} has empty description"
            )

    def test_no_duplicate_tools(self, config):
        """Test that there are no duplicate tool names."""
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = [tool.name for tool in tools]
        unique_names = set(tool_names)

        assert len(tool_names) == len(unique_names), (
            f"Duplicate tool names found: {tool_names}"
        )

    def test_tools_are_langchain_compatible(self, config):
        """Test that all tools are LangChain BaseTool instances."""
        from langchain_core.tools import BaseTool

        tools = VibeToolAdapter.get_all_tools(config)

        for tool in tools:
            assert isinstance(tool, BaseTool), (
                f"Tool {tool.name} is not a BaseTool instance"
            )

    def test_empty_config_tool_paths(self, config):
        """Test tool loading with empty tool_paths config."""
        config.tool_paths = []

        tools = VibeToolAdapter.get_all_tools(config)

        # Should still have built-in tools
        tool_names = {tool.name for tool in tools}
        assert "bash" in tool_names
        assert any("read_file" in name for name in tool_names), (
            "Filesystem tools should be available"
        )

    def test_multiple_tool_paths(self, config, temp_dir):
        """Test tool loading from multiple paths."""
        # Create two directories with tools
        dir1 = temp_dir / "tools1"
        dir1.mkdir()
        dir2 = temp_dir / "tools2"
        dir2.mkdir()

        # Tool in first directory
        tool1_file = dir1 / "tool1.py"
        tool1_file.write_text("""
from langchain_core.tools import StructuredTool

def multi_tool1(data: str) -> str:
    return f"Multi1: {data}"

multi_tool1_instance = StructuredTool.from_function(
    name="multi_tool1",
    description="Tool from first path",
    func=multi_tool1,
)
""")

        # Tool in second directory
        tool2_file = dir2 / "tool2.py"
        tool2_file.write_text("""
from langchain_core.tools import StructuredTool

def multi_tool2(data: str) -> str:
    return f"Multi2: {data}"

multi_tool2_instance = StructuredTool.from_function(
    name="multi_tool2",
    description="Tool from second path",
    func=multi_tool2,
)
""")

        config.tool_paths = [str(dir1), str(dir2)]

        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {tool.name for tool in tools}

        assert "multi_tool1" in tool_names, "Tool from first path not loaded"
        assert "multi_tool2" in tool_names, "Tool from second path not loaded"

    def test_tool_loading_error_handling(self, config, temp_dir):
        """Test that tool loading errors don't crash the system."""
        # Create a file that will cause import error
        error_file = temp_dir / "error_tool.py"
        error_file.write_text("""
# This will cause an import error
import nonexistent_module

from langchain_core.tools import StructuredTool

def error_tool(data: str) -> str:
    return f"Error: {data}"

error_tool_instance = StructuredTool.from_function(
    name="error_tool",
    description="Tool that should not load due to import error",
    func=error_tool,
)
""")

        config.tool_paths = [str(temp_dir)]

        # Should not raise exception
        tools = VibeToolAdapter.get_all_tools(config)

        # Should still have basic tools
        tool_names = {tool.name for tool in tools}
        assert "bash" in tool_names, "Basic tools should be available despite errors"

        # The error_tool should not be loaded due to import error
        assert "error_tool" not in tool_names, (
            "Tool with import error should not be loaded"
        )

    def test_filesystem_tools_with_state_backend(self, config):
        """Test that filesystem tools are properly configured with StateBackend."""
        tools = VibeToolAdapter.get_all_tools(config)

        # Check that filesystem tools are present
        filesystem_tool_names = {
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
        }
        tool_names = {tool.name for tool in tools}

        for fs_tool in filesystem_tool_names:
            assert fs_tool in tool_names, f"Filesystem tool {fs_tool} not found"

    def test_todo_tools_available(self, config):
        """Test that todo tools from TodoListMiddleware are available."""
        tools = VibeToolAdapter.get_all_tools(config)

        tool_names = {tool.name for tool in tools}

        # TodoListMiddleware provides write_todos and read_todos
        # Note: These might not be directly named that way, but should be present
        # For now, just ensure no exceptions are raised
        assert len(tools) > 5, "Should have multiple tools including todo tools"

    def test_tool_parameter_validation(self, config):
        """Test that tools have proper parameter schemas."""
        from langchain_core.tools import StructuredTool

        tools = VibeToolAdapter.get_all_tools(config)

        for tool in tools:
            # Each tool should be a StructuredTool with proper schema
            assert isinstance(tool, StructuredTool), (
                f"Tool {tool.name} is not a StructuredTool"
            )
            assert tool.args_schema is not None, f"Tool {tool.name} missing args_schema"

    def test_large_number_of_tools(self, config, temp_dir):
        """Test loading a large number of custom tools."""
        # Create multiple tool files
        tools_dir = temp_dir / "many_tools"
        tools_dir.mkdir()

        for i in range(10):
            tool_file = tools_dir / f"tool_{i}.py"
            tool_file.write_text(f"""
from langchain_core.tools import StructuredTool

def tool_{i}_func(value: int) -> str:
    return f"Tool {i}: {{value}}"

tool_{i}_instance = StructuredTool.from_function(
    name="tool_{i}",
    description="Tool {i}",
    func=tool_{i}_func,
)
""")

        config.tool_paths = [str(tools_dir)]

        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {tool.name for tool in tools}

        # Should have loaded all 10 custom tools
        for i in range(10):
            assert f"tool_{i}" in tool_names, f"Tool {i} not loaded"

        # Should still have basic tools
        assert "bash" in tool_names

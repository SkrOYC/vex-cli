"""Unit tests for VibeToolAdapter with DeepAgents integration."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from vibe.core.config import VibeConfig
from vibe.core.engine.tools import VibeToolAdapter


class TestVibeToolAdapter:
    """Test VibeToolAdapter functionality."""

    def test_get_all_tools_returns_sequence(self, deepagents_config: VibeConfig):
        """Test get_all_tools returns a sequence of tools."""
        from langchain_core.tools import BaseTool

        tools = VibeToolAdapter.get_all_tools(deepagents_config)

        assert hasattr(tools, "__iter__")  # It's iterable
        for tool in tools:
            assert isinstance(tool, BaseTool)
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "args_schema")
            assert hasattr(tool, "run") or hasattr(tool, "ainvoke")

    def test_get_all_tools_includes_bash(self, deepagents_config: VibeConfig):
        """Test get_all_tools includes the bash tool."""
        tools = VibeToolAdapter.get_all_tools(deepagents_config)
        bash_tools = [tool for tool in tools if tool.name == "bash"]

        assert len(bash_tools) >= 1  # At least one bash tool should be present
        bash_tool = bash_tools[0]
        assert "bash" in bash_tool.name
        assert "Execute a bash command" in bash_tool.description

    def test_create_bash_tool(self, deepagents_config: VibeConfig):
        """Test bash tool creation."""
        bash_tool = VibeToolAdapter._create_bash_tool(deepagents_config)

        assert isinstance(bash_tool, BaseTool)
        assert bash_tool.name == "bash"
        assert "Execute a bash command" in bash_tool.description

    def test_load_custom_tools_empty(self):
        """Test loading custom tools from empty path."""
        tools = VibeToolAdapter._load_custom_tools("")

        assert isinstance(tools, list)
        assert len(tools) == 0

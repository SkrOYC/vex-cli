"""Tests for VibeToolAdapter."""

import pytest
from vibe.core.engine.tools import VibeToolAdapter
from vibe.core.config import VibeConfig


class TestVibeToolAdapter:
    def test_get_all_tools(self):
        """Test getting all tools."""
        config = VibeConfig()  # Minimal config
        tools = VibeToolAdapter.get_all_tools(config)
        assert isinstance(tools, list)
        # Should have at least the bash tool
        assert len(tools) >= 1

    def test_create_bash_tool(self):
        """Test bash tool creation."""
        config = VibeConfig()
        tool = VibeToolAdapter._create_bash_tool(config)
        assert tool.name == "bash"
        assert "bash command" in tool.description.lower()
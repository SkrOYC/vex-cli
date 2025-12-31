"""Tests for VibeToolAdapter tool filtering functionality."""

from __future__ import annotations

from langchain_core.tools import BaseTool
import pytest

from vibe.core.config import VibeConfig
from vibe.core.engine.tools import VibeToolAdapter


class TestMatchesToolPattern:
    """Tests for _matches_tool_pattern method."""

    def test_exact_match(self):
        """Test exact name matching."""
        assert VibeToolAdapter._matches_tool_pattern("bash", "bash") is True
        assert VibeToolAdapter._matches_tool_pattern("read", "bash") is False
        assert VibeToolAdapter._matches_tool_pattern("create", "create") is True

    def test_glob_pattern(self):
        """Test glob pattern matching."""
        assert VibeToolAdapter._matches_tool_pattern("bash", "bash*") is True
        assert VibeToolAdapter._matches_tool_pattern("bash_tool", "bash*") is True
        assert VibeToolAdapter._matches_tool_pattern("read", "bash*") is False
        assert VibeToolAdapter._matches_tool_pattern("read_file", "read*") is True
        assert VibeToolAdapter._matches_tool_pattern("edit_file", "*file") is True

    def test_regex_pattern(self):
        """Test regex pattern matching with re: prefix."""
        assert VibeToolAdapter._matches_tool_pattern("bash", "re:^bash$") is True
        assert VibeToolAdapter._matches_tool_pattern("bash_tool", "re:^bash.*") is True
        assert VibeToolAdapter._matches_tool_pattern("read_file", "re:.*_file$") is True
        assert VibeToolAdapter._matches_tool_pattern("bash", "re:^read.*") is False

    def test_empty_pattern(self):
        """Test empty pattern returns False."""
        assert VibeToolAdapter._matches_tool_pattern("bash", "") is False

    def test_invalid_regex(self):
        """Test invalid regex pattern returns False."""
        assert VibeToolAdapter._matches_tool_pattern("bash", "re:[invalid") is False


class TestApplyToolFiltering:
    """Tests for _apply_tool_filtering method."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""
        tools: list[BaseTool] = []

        class MockTool(BaseTool):
            def __init__(self, name: str):
                super().__init__(name=name, description=f"{name} tool")

            def _run(self, **kwargs):
                return f"Ran {self.name}"

        for name in ["bash", "create", "read", "edit", "edit_file", "list", "grep"]:
            tools.append(MockTool(name))
        return tools

    def test_no_filtering_when_empty(self, mock_tools):
        """Test all tools returned when no filtering config."""
        config = VibeConfig(enabled_tools=[], disabled_tools=[])
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == len(mock_tools)
        assert set(t.name for t in result) == set(t.name for t in mock_tools)

    def test_no_filtering_when_none(self, mock_tools):
        """Test all tools returned when config is None."""
        config = VibeConfig()
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == len(mock_tools)

    def test_whitelist_exact_match(self, mock_tools):
        """Test whitelist with exact matches."""
        config = VibeConfig(enabled_tools=["bash", "read"])
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == 2
        assert set(t.name for t in result) == {"bash", "read"}

    def test_whitelist_glob_pattern(self, mock_tools):
        """Test whitelist with glob patterns."""
        config = VibeConfig(enabled_tools=["edit*"])
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == 2
        assert set(t.name for t in result) == {"edit", "edit_file"}

    def test_whitelist_regex_pattern(self, mock_tools):
        """Test whitelist with regex patterns."""
        config = VibeConfig(enabled_tools=["re:^c.*"])
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == 1
        assert result[0].name == "create"

    def test_blacklist_exact_match(self, mock_tools):
        """Test blacklist with exact matches."""
        config = VibeConfig(disabled_tools=["bash", "grep"])
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == len(mock_tools) - 2
        assert "bash" not in set(t.name for t in result)
        assert "grep" not in set(t.name for t in result)

    def test_blacklist_glob_pattern(self, mock_tools):
        """Test blacklist with glob patterns."""
        config = VibeConfig(disabled_tools=["edit*"])
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == len(mock_tools) - 2
        assert "edit" not in set(t.name for t in result)
        assert "edit_file" not in set(t.name for t in result)

    def test_blacklist_regex_pattern(self, mock_tools):
        """Test blacklist with regex patterns."""
        config = VibeConfig(disabled_tools=["re:.*_file$"])
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == len(mock_tools) - 1
        assert "edit_file" not in set(t.name for t in result)

    def test_blacklist_overrides_whitelist(self, mock_tools):
        """Test blacklist takes precedence over whitelist."""
        # Whitelist allows all, blacklist removes some
        config = VibeConfig(
            enabled_tools=[
                "bash",
                "create",
                "read",
                "edit",
                "edit_file",
                "list",
                "grep",
            ],
            disabled_tools=["bash", "grep"],
        )
        result = VibeToolAdapter._apply_tool_filtering(mock_tools, config)
        assert len(result) == len(mock_tools) - 2
        assert "bash" not in set(t.name for t in result)
        assert "grep" not in set(t.name for t in result)


class TestGetAllTools:
    """Tests for get_all_tools method."""

    def test_get_all_tools_returns_sequence(self):
        """Test get_all_tools returns a Sequence."""
        config = VibeConfig()
        tools = VibeToolAdapter.get_all_tools(config)
        assert isinstance(tools, (list, tuple))

    def test_get_all_tools_includes_bash(self):
        """Test bash tool is included."""
        config = VibeConfig()
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}
        assert "bash" in tool_names

    def test_get_all_tools_includes_filesystem_tools(self):
        """Test all filesystem tools are included by default."""
        config = VibeConfig()
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}
        expected_filesystem = {
            "create",
            "read_file",
            "edit",
            "edit_file",
            "list_files",
            "grep",
            "insert_line",
        }
        assert expected_filesystem.issubset(tool_names)

    def test_tool_filtering_whitelist(self):
        """Test tool filtering with whitelist."""
        config = VibeConfig(enabled_tools=["bash", "read_file"])
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}
        assert tool_names == {"bash", "read_file"}

    def test_tool_filtering_blacklist(self):
        """Test tool filtering with blacklist."""
        config = VibeConfig(disabled_tools=["bash", "grep"])
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}
        assert "bash" not in tool_names
        assert "grep" not in tool_names
        # Other tools should still be present
        assert "read_file" in tool_names

    def test_glob_pattern_filtering(self):
        """Test glob pattern filtering."""
        config = VibeConfig(enabled_tools=["edit*"])
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}
        assert "edit" in tool_names
        assert "edit_file" in tool_names
        assert "bash" not in tool_names
        assert "read_file" not in tool_names

    def test_regex_pattern_filtering(self):
        """Test regex pattern filtering."""
        config = VibeConfig(disabled_tools=["re:^edit_file$"])
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}
        assert "edit_file" not in tool_names
        assert "bash" in tool_names
        assert "read_file" in tool_names
        assert "create" in tool_names

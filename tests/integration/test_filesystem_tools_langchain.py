"""Integration tests for filesystem tools with LangChain engine.

These tests verify that filesystem tools work correctly with the LangChain engine
and are properly integrated into the VibeToolAdapter.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from vibe.core.config import VibeConfig
from vibe.core.engine.tools import VibeToolAdapter


class TestFilesystemToolsIntegration:
    """Integration tests for filesystem tools."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_create_tool_integration(self, temp_dir):
        """Test CreateTool creates files correctly."""
        from vibe.core.tools.filesystem.create import CreateTool

        tool = CreateTool(workdir=temp_dir)
        result = await tool._arun(path="test.txt", file_text="Hello, World!")

        assert "created successfully" in result.output.lower()
        assert (temp_dir / "test.txt").exists()
        assert (temp_dir / "test.txt").read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_file_tool_integration(self, temp_dir):
        """Test ReadFileTool reads files correctly."""
        from vibe.core.tools.filesystem.read_file import ReadFileTool

        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        tool = ReadFileTool(workdir=temp_dir)
        result = await tool._arun(path="test.txt", view_type="content")

        assert result.content is not None
        assert "Hello, World!" in result.content.output

    @pytest.mark.asyncio
    async def test_edit_file_tool_integration(self, temp_dir):
        """Test EditFileTool performs string replacement correctly."""
        from vibe.core.tools.filesystem.edit_file import EditFileTool

        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        tool = EditFileTool(workdir=temp_dir)
        result = await tool._arun(path="test.txt", old_str="World", new_str="Vibe")

        assert "successfully" in result.output.lower()
        assert (temp_dir / "test.txt").read_text() == "Hello, Vibe!"

    @pytest.mark.asyncio
    async def test_list_files_tool_integration(self, temp_dir):
        """Test ListFilesTool lists files correctly."""
        from vibe.core.tools.filesystem.list_files import ListFilesTool

        # Create test files
        (temp_dir / "test1.txt").write_text("test1")
        (temp_dir / "test2.txt").write_text("test2")

        tool = ListFilesTool(workdir=temp_dir)
        result = await tool._arun(path=".", patterns=["*.txt"])

        assert "test1.txt" in result.output
        assert "test2.txt" in result.output

    @pytest.mark.asyncio
    async def test_grep_tool_integration(self, temp_dir):
        """Test GrepTool searches files correctly."""
        from vibe.core.tools.filesystem.grep import GrepTool

        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World! Hello, Vibe!")

        tool = GrepTool(workdir=temp_dir)
        result = await tool._arun(path=".", query="Hello")

        assert "Hello" in result.output
        assert "test.txt" in result.output

    @pytest.mark.asyncio
    async def test_insert_line_tool_integration(self, temp_dir):
        """Test InsertLineTool inserts lines correctly."""
        from vibe.core.tools.filesystem.insert_line import InsertLineTool

        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("line 1\nline 3")

        tool = InsertLineTool(workdir=temp_dir)
        result = await tool._arun(
            path="test.txt",
            new_str="line 2",
            insert_line=2,
        )

        assert "inserted" in result.output.lower()
        content = (temp_dir / "test.txt").read_text()
        assert "line 1" in content
        assert "line 2" in content
        assert "line 3" in content


class TestVibeToolAdapterFilesystemIntegration:
    """Tests for VibeToolAdapter filesystem tool integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_filesystem_tools_in_adapter(self, temp_dir):
        """Test that filesystem tools are included in VibeToolAdapter."""
        config = VibeConfig(workdir=temp_dir)
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}

        # Check all filesystem tools are present
        expected_tools = {
            "bash",
            "create",
            "read_file",
            "edit",
            "edit_file",
            "list_files",
            "grep",
            "insert_line",
        }
        assert expected_tools.issubset(tool_names), (
            f"Missing tools: {expected_tools - tool_names}"
        )

    @pytest.mark.asyncio
    async def test_filesystem_tools_functional(self, temp_dir):
        """Test that filesystem tools from adapter are functional."""
        config = VibeConfig(workdir=temp_dir)
        tools = VibeToolAdapter.get_all_tools(config)
        tools_dict = {t.name: t for t in tools}

        # Test CreateTool
        create_tool = tools_dict["create"]
        result = await create_tool._arun(path="test.txt", file_text="Hello, World!")
        assert "created successfully" in result.output.lower()

        # Test ReadFileTool
        read_tool = tools_dict["read_file"]
        result = await read_tool._arun(path="test.txt", view_type="content")
        assert result.content is not None
        assert "Hello, World!" in result.content.output

        # Test EditFileTool
        edit_tool = tools_dict["edit_file"]
        result = await edit_tool._arun(path="test.txt", old_str="World", new_str="Vibe")
        assert "successfully" in result.output.lower()

        # Verify file was edited
        content = (temp_dir / "test.txt").read_text()
        assert "Hello, Vibe!" in content


class TestToolFilteringIntegration:
    """Integration tests for tool filtering with filesystem tools."""

    def test_whitelist_filesystem_tools(self):
        """Test whitelist filtering includes only specified filesystem tools."""
        config = VibeConfig(enabled_tools=["create", "read_file"])
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}

        assert tool_names == {"create", "read_file"}

    def test_blacklist_filesystem_tools(self):
        """Test blacklist filtering excludes specified filesystem tools."""
        config = VibeConfig(disabled_tools=["bash", "grep"])
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}

        assert "bash" not in tool_names
        assert "grep" not in tool_names
        # Other tools should still be present
        assert "create" in tool_names
        assert "read_file" in tool_names
        assert "edit_file" in tool_names

    def test_glob_pattern_filesystem_tools(self):
        """Test glob pattern filtering with filesystem tools."""
        config = VibeConfig(enabled_tools=["edit*"])
        tools = VibeToolAdapter.get_all_tools(config)
        tool_names = {t.name for t in tools}

        assert "edit" in tool_names
        assert "edit_file" in tool_names
        assert "bash" not in tool_names
        assert "read_file" not in tool_names

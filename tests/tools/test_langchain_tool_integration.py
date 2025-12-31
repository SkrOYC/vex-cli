"""Integration tests for LangChain BaseTool compatibility.

Tests verify that all migrated tools work correctly with LangChain agents
and follow LangChain BaseTool conventions.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from langchain_core.tools import BaseTool
import pytest

from vibe.core.tools.filesystem.create import CreateTool
from vibe.core.tools.filesystem.edit_file import EditFileTool
from vibe.core.tools.filesystem.grep import GrepTool
from vibe.core.tools.filesystem.insert_line import InsertLineTool
from vibe.core.tools.filesystem.list_files import ListFilesTool
from vibe.core.tools.filesystem.read_file import ReadFileTool


class TestToolLangChainCompatibility:
    """Test that tools are LangChain BaseTool compatible."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.parametrize(
        "tool_class",
        [
            ReadFileTool,
            EditFileTool,
            CreateTool,
            ListFilesTool,
            GrepTool,
            InsertLineTool,
        ],
    )
    def test_tool_is_langchain_basetool(self, temp_dir, tool_class):
        """Test that tools are LangChain BaseTool compatible."""
        tool = tool_class(workdir=temp_dir)
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "args_schema")
        assert hasattr(tool, "_arun")

    @pytest.mark.asyncio
    async def test_read_file_tool_async_execution(self, temp_dir):
        """Test ReadFileTool executes correctly via _arun."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content", encoding="utf-8")

        tool = ReadFileTool(workdir=temp_dir)
        result = await tool._arun(path=str(test_file), view_type="content")

        assert result.content is not None
        assert "test content" in result.content.output

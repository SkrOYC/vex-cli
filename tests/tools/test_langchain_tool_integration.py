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

    def test_read_file_tool_is_langchain_basetool(self, temp_dir):
        """Test ReadFileTool is a LangChain BaseTool."""
        tool = ReadFileTool(workdir=temp_dir)
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "args_schema")
        assert hasattr(tool, "_arun")

    def test_edit_file_tool_is_langchain_basetool(self, temp_dir):
        """Test EditFileTool is a LangChain BaseTool."""
        tool = EditFileTool(workdir=temp_dir)
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "args_schema")
        assert hasattr(tool, "_arun")

    def test_create_tool_is_langchain_basetool(self, temp_dir):
        """Test CreateTool is a LangChain BaseTool."""
        tool = CreateTool(workdir=temp_dir)
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "args_schema")
        assert hasattr(tool, "_arun")

    def test_list_files_tool_is_langchain_basetool(self, temp_dir):
        """Test ListFilesTool is a LangChain BaseTool."""
        tool = ListFilesTool(workdir=temp_dir)
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "args_schema")
        assert hasattr(tool, "_arun")

    def test_grep_tool_is_langchain_basetool(self, temp_dir):
        """Test GrepTool is a LangChain BaseTool."""
        tool = GrepTool(workdir=temp_dir)
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "args_schema")
        assert hasattr(tool, "_arun")

    def test_insert_line_tool_is_langchain_basetool(self, temp_dir):
        """Test InsertLineTool is a LangChain BaseTool."""
        tool = InsertLineTool(workdir=temp_dir)
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
        result = await tool._arun(path=str(test_file))

        assert "test content" in result.content.output

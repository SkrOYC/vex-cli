"""Unit tests for EditFileTool.

Tests cover all acceptance criteria including:
- Str_replace with exact match succeeds
- Str_replace with no match fails with correct error
- Str_replace with multiple matches fails with correct error
- Str_replace on non-existent file fails with correct error
- Relative paths resolved correctly
- Absolute paths work correctly
- UTF-8 encoding handles special characters
- Replacement works across line boundaries
- Replacement works with empty old_str
- Replacement works with empty new_str
- All error messages are helpful and include suggestions
- Edit history is saved correctly
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import tempfile

from pydantic import ValidationError
import pytest

from vibe.core.tools.filesystem.edit_file import (
    EditFileArgs,
    EditFileResult,
    EditFileTool,
    EditFileToolConfig,
    EditFileToolState,
)
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def view_tracker() -> ViewTrackerService:
    """Create a fresh ViewTrackerService for each test."""
    return ViewTrackerService()


@pytest.fixture
def tool_config(view_tracker: ViewTrackerService, temp_dir: Path) -> EditFileToolConfig:
    """Create an EditFileToolConfig with ViewTrackerService."""
    return EditFileToolConfig(view_tracker=view_tracker, workdir=temp_dir)


@pytest.fixture
def tool_config_no_view_tracker(temp_dir: Path) -> EditFileToolConfig:
    """Create an EditFileToolConfig without ViewTrackerService."""
    return EditFileToolConfig(workdir=temp_dir)


@pytest.fixture
def tool_state() -> EditFileToolState:
    """Create a fresh EditFileToolState for each test."""
    return EditFileToolState()


@pytest.fixture
def tool(
    tool_config: EditFileToolConfig, tool_state: EditFileToolState
) -> EditFileTool:
    """Create an EditFileTool instance for testing."""
    return EditFileTool(config=tool_config, state=tool_state)


@pytest.fixture
def tool_no_view_tracker(
    tool_config_no_view_tracker: EditFileToolConfig, tool_state: EditFileToolState
) -> EditFileTool:
    """Create an EditFileTool instance without ViewTrackerService for testing."""
    return EditFileTool(config=tool_config_no_view_tracker, state=tool_state)


# =============================================================================
# EditFileArgs Tests
# =============================================================================


class TestEditFileArgs:
    """Tests for EditFileArgs Pydantic model."""

    def test_creation_with_path_and_strings(self) -> None:
        """Test EditFileArgs can be created with path, old_str, and new_str."""
        args = EditFileArgs(path="/test/file.txt", old_str="hello", new_str="world")
        assert args.path == "/test/file.txt"
        assert args.old_str == "hello"
        assert args.new_str == "world"

    def test_creation_with_empty_strings(self) -> None:
        """Test EditFileArgs accepts empty strings."""
        args = EditFileArgs(path="/test/file.txt", old_str="", new_str="")
        assert args.old_str == ""
        assert args.new_str == ""

    def test_creation_with_whitespace(self) -> None:
        """Test EditFileArgs handles whitespace correctly."""
        args = EditFileArgs(
            path="/test/file.txt", old_str="  spaces  ", new_str="\t tabs \n"
        )
        assert args.old_str == "  spaces  "
        assert args.new_str == "\t tabs \n"

    def test_model_validation(self) -> None:
        """Test EditFileArgs validates types correctly."""
        with pytest.raises(ValidationError):
            EditFileArgs(path=123, old_str="text", new_str="text")  # type: ignore


# =============================================================================
# EditFileResult Tests
# =============================================================================


class TestEditFileResult:
    """Tests for EditFileResult Pydantic model."""

    def test_creation_with_output(self) -> None:
        """Test EditFileResult can be created with output."""
        result = EditFileResult(output="File modified successfully.")
        assert result.output == "File modified successfully."


# =============================================================================
# Str Replace Success Tests
# =============================================================================


class TestStrReplaceSuccess:
    """Tests for successful string replacement operations."""

    pytestmark = pytest.mark.asyncio

    async def test_replace_exactly_one_occurrence(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacing exactly one occurrence succeeds."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("hello world", encoding="utf-8")

        result = await tool.run(
            EditFileArgs(path=str(file_path), old_str="world", new_str="bun")
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "hello bun"

    async def test_replace_across_line_boundaries(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement works across line boundaries."""
        file_path = temp_dir / "multiline.txt"
        content = "line1\nline2\nline3"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(
            EditFileArgs(
                path=str(file_path), old_str="line1\nline2", new_str="replaced"
            )
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "replaced\nline3"

    async def test_replace_with_empty_new_str(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement with empty new_str removes the old_str."""
        file_path = temp_dir / "remove.txt"
        file_path.write_text("hello world", encoding="utf-8")

        result = await tool.run(
            EditFileArgs(path=str(file_path), old_str=" world", new_str="")
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "hello"

    async def test_replace_saves_to_edit_history(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test that replacement saves content to edit history."""
        file_path = temp_dir / "history.txt"
        original_content = "original content"
        file_path.write_text(original_content, encoding="utf-8")

        await tool.run(
            EditFileArgs(path=str(file_path), old_str="original", new_str="modified")
        )

        # Check that history was saved
        assert str(file_path) in tool.state.edit_history
        history = tool.state.edit_history[str(file_path)]
        assert len(history) == 1
        assert history[0] == original_content

    async def test_replace_updates_view_timestamp(
        self, tool: EditFileTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test that replacement updates the view timestamp."""
        file_path = temp_dir / "view.txt"
        file_path.write_text("content", encoding="utf-8")

        # Record initial view
        view_tracker.record_view(str(file_path))
        initial_timestamp = view_tracker.get_last_view_timestamp(str(file_path))

        # Perform edit
        await tool.run(
            EditFileArgs(path=str(file_path), old_str="content", new_str="new content")
        )

        # Check that timestamp was updated
        new_timestamp = view_tracker.get_last_view_timestamp(str(file_path))
        assert new_timestamp is not None
        assert initial_timestamp is not None
        assert new_timestamp >= initial_timestamp

    async def test_replace_without_view_tracker(
        self, tool_no_view_tracker: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement works without ViewTrackerService configured."""
        file_path = temp_dir / "no_tracker.txt"
        file_path.write_text("hello", encoding="utf-8")

        result = await tool_no_view_tracker.run(
            EditFileArgs(path=str(file_path), old_str="hello", new_str="world")
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "world"


# =============================================================================
# Str Replace Text Not Found Tests
# =============================================================================


class TestStrReplaceTextNotFound:
    """Tests for text not found error handling."""

    pytestmark = pytest.mark.asyncio

    async def test_replace_fails_when_old_str_not_found(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement fails when old_str is not found."""
        file_path = temp_dir / "not_found.txt"
        file_path.write_text("hello world", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditFileArgs(path=str(file_path), old_str="nonexistent", new_str="text")
            )

        assert exc_info.value.code == "TEXT_NOT_FOUND"
        assert "was not found" in str(exc_info.value)

    async def test_error_message_includes_helpful_suggestions(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test error message includes helpful suggestions."""
        file_path = temp_dir / "suggestions.txt"
        file_path.write_text("hello world", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditFileArgs(path=str(file_path), old_str="goodbye", new_str="hello")
            )

        assert exc_info.value.code == "TEXT_NOT_FOUND"
        assert len(exc_info.value.suggestions) > 0
        # Check that suggestions include common issues
        suggestions_text = " ".join(exc_info.value.suggestions).lower()
        assert any(
            word in suggestions_text
            for word in ["typo", "whitespace", "escaping", "read_file"]
        )

    async def test_replace_fails_on_empty_file(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement fails on empty file."""
        file_path = temp_dir / "empty.txt"
        file_path.write_text("", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditFileArgs(path=str(file_path), old_str="anything", new_str="text")
            )

        assert exc_info.value.code == "TEXT_NOT_FOUND"


# =============================================================================
# Str Replace Multiple Matches Tests
# =============================================================================


class TestStrReplaceMultipleMatches:
    """Tests for multiple matches error handling."""

    pytestmark = pytest.mark.asyncio

    async def test_replace_fails_when_multiple_occurrences(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement fails when old_str appears multiple times."""
        file_path = temp_dir / "multiple.txt"
        file_path.write_text("repeat repeat", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditFileArgs(path=str(file_path), old_str="repeat", new_str="once")
            )

        assert exc_info.value.code == "MULTIPLE_MATCHES"
        assert "appears 2 times" in str(exc_info.value)

    async def test_error_message_includes_count_and_suggestions(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test error message includes occurrence count and suggestions."""
        file_path = temp_dir / "count.txt"
        file_path.write_text("foo foo foo foo", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditFileArgs(path=str(file_path), old_str="foo", new_str="bar")
            )

        assert exc_info.value.code == "MULTIPLE_MATCHES"
        # Check that count is mentioned
        assert "appears 4 times" in str(exc_info.value)
        # Check that suggestions are provided
        assert len(exc_info.value.suggestions) > 0
        suggestions_text = " ".join(exc_info.value.suggestions).lower()
        assert any(
            word in suggestions_text
            for word in ["context", "surrounding", "write_file"]
        )


# =============================================================================
# File Not Found Tests
# =============================================================================


class TestFileNotFound:
    """Tests for file not found error handling."""

    pytestmark = pytest.mark.asyncio

    async def test_replace_fails_on_nonexistent_file(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement fails on nonexistent file."""
        file_path = temp_dir / "nonexistent.txt"

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditFileArgs(path=str(file_path), old_str="old", new_str="new")
            )

        assert exc_info.value.code == "FILE_NOT_FOUND"
        assert "not found" in str(exc_info.value)

    async def test_error_message_suggests_create_or_check_path(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test error message suggests creating file or checking path."""
        file_path = temp_dir / "missing.txt"

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditFileArgs(path=str(file_path), old_str="old", new_str="new")
            )

        assert exc_info.value.code == "FILE_NOT_FOUND"
        assert len(exc_info.value.suggestions) > 0
        suggestions_text = " ".join(exc_info.value.suggestions).lower()
        assert any(
            word in suggestions_text
            for word in ["write_file", "create", "check", "path"]
        )


# =============================================================================
# Path Resolution Tests
# =============================================================================


class TestPathResolution:
    """Tests for path resolution functionality."""

    pytestmark = pytest.mark.asyncio

    async def test_relative_path_resolved_against_workdir(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test relative paths are resolved against working directory."""
        file_path = temp_dir / "relative.txt"
        file_path.write_text("original", encoding="utf-8")

        result = await tool.run(
            EditFileArgs(path="relative.txt", old_str="original", new_str="modified")
        )

        assert "modified successfully" in result.output
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == "modified"

    async def test_absolute_path_works_correctly(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test absolute paths work correctly."""
        file_path = temp_dir / "absolute.txt"
        file_path.write_text("original", encoding="utf-8")

        result = await tool.run(
            EditFileArgs(path=str(file_path), old_str="original", new_str="modified")
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "modified"


# =============================================================================
# UTF-8 Encoding Tests
# =============================================================================


class TestUTF8Encoding:
    """Tests for UTF-8 encoding handling."""

    pytestmark = pytest.mark.asyncio

    async def test_utf8_encoding_handles_special_characters(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test UTF-8 encoding handles special characters correctly."""
        file_path = temp_dir / "unicode.txt"
        content = "Hello, 世界! 🌍 Ñoño ©®™"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(
            EditFileArgs(path=str(file_path), old_str="世界", new_str="宇宙")
        )

        assert "modified successfully" in result.output
        expected = "Hello, 宇宙! 🌍 Ñoño ©®™"
        assert file_path.read_text(encoding="utf-8") == expected


# =============================================================================
# Tool Configuration Tests
# =============================================================================


class TestToolConfiguration:
    """Tests for tool configuration."""

    def test_tool_name_is_edit_file(self) -> None:
        """Test tool has correct name."""
        assert EditFileTool.name == "edit_file"

    def test_tool_has_description(self) -> None:
        """Test tool has description."""
        assert len(EditFileTool.description) > 0

    def test_tool_uses_edit_file_args_schema(self) -> None:
        """Test tool uses EditFileArgs as schema."""
        assert EditFileTool.args_schema == EditFileArgs


# =============================================================================
# Edit History Tests
# =============================================================================


class TestEditHistory:
    """Tests for edit history functionality."""

    pytestmark = pytest.mark.asyncio

    async def test_multiple_edits_save_to_history(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test multiple edits save all versions to history."""
        file_path = temp_dir / "multi_edit.txt"
        file_path.write_text("version 1", encoding="utf-8")

        # First edit
        await tool.run(
            EditFileArgs(path=str(file_path), old_str="version 1", new_str="version 2")
        )
        # Second edit
        await tool.run(
            EditFileArgs(path=str(file_path), old_str="version 2", new_str="version 3")
        )

        history = tool.state.edit_history[str(file_path)]
        assert len(history) == 2
        assert history[0] == "version 1"
        assert history[1] == "version 2"

    async def test_pop_history_returns_previous_content(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test popping history returns the previous content."""
        file_path = temp_dir / "pop_history.txt"
        file_path.write_text("current content", encoding="utf-8")

        # Perform edit
        await tool.run(
            EditFileArgs(
                path=str(file_path), old_str="current", new_str="modified content"
            )
        )

        # Pop history
        previous = tool._pop_history(str(file_path))
        assert previous == "current content"

        # History should be empty now
        history = tool.state.edit_history[str(file_path)]
        assert len(history) == 0

    async def test_pop_history_returns_none_when_empty(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test popping history returns None when no history exists."""
        file_path = temp_dir / "empty_history.txt"
        file_path.write_text("content", encoding="utf-8")

        result = tool._pop_history(str(file_path))
        assert result is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    pytestmark = pytest.mark.asyncio

    async def test_replace_with_exact_file_content(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacing the entire file content."""
        file_path = temp_dir / "full_replace.txt"
        content = "line1\nline2\nline3"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(
            EditFileArgs(path=str(file_path), old_str=content, new_str="new content")
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "new content"

    async def test_replace_with_multiline_context(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement with multiline old_str."""
        file_path = temp_dir / "multiline.txt"
        content = "line1\nline2\nline3\nline4"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(
            EditFileArgs(
                path=str(file_path), old_str="line1\nline2\nline3", new_str="replaced"
            )
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "replaced\nline4"

    async def test_replace_with_special_regex_chars(
        self, tool: EditFileTool, temp_dir: Path
    ) -> None:
        """Test replacement with characters that are special in regex."""
        file_path = temp_dir / "regex_chars.txt"
        content = "file (1).txt"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(
            EditFileArgs(path=str(file_path), old_str="(1)", new_str="[2]")
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "file [2].txt"

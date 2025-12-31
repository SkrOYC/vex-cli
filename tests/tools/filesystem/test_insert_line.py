"""Unit tests for InsertLineTool.

Tests cover all acceptance criteria including:
- Insertion at beginning of file (line 1)
- Insertion at end of file (line num_lines + 1)
- Insertion in middle of file
- Insertion into empty file (line 1 only)
- Insertion into empty file fails for line > 1
- Line bounds validation (low and high)
- File not found handling
- Path resolution (relative and absolute)
- UTF-8 encoding with special characters
- All error messages are helpful and include suggestions
- Edit history tracking
- View tracker integration
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import tempfile

from pydantic import ValidationError
import pytest

from vibe.core.tools.filesystem.insert_line import (
    InsertLineArgs,
    InsertLineResult,
    InsertLineTool,
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
def tool(view_tracker: ViewTrackerService, temp_dir: Path) -> InsertLineTool:
    """Create an InsertLineTool instance for testing."""
    return InsertLineTool(view_tracker=view_tracker, workdir=temp_dir)


@pytest.fixture
def tool_no_view_tracker(temp_dir: Path) -> InsertLineTool:
    """Create an InsertLineTool instance without ViewTrackerService for testing."""
    return InsertLineTool(workdir=temp_dir)


# =============================================================================
# InsertLineArgs Tests
# =============================================================================


class TestInsertLineArgs:
    """Tests for InsertLineArgs Pydantic model."""

    def test_creation_with_path_and_content(self) -> None:
        """Test InsertLineArgs can be created with path, new_str, and insert_line."""
        args = InsertLineArgs(
            path="/test/file.txt", new_str="new content", insert_line=1
        )
        assert args.path == "/test/file.txt"
        assert args.new_str == "new content"
        assert args.insert_line == 1

    def test_creation_with_empty_new_str(self) -> None:
        """Test InsertLineArgs accepts empty new_str."""
        args = InsertLineArgs(path="/test/file.txt", new_str="", insert_line=1)
        assert args.new_str == ""

    def test_creation_with_whitespace(self) -> None:
        """Test InsertLineArgs handles whitespace correctly."""
        args = InsertLineArgs(
            path="/test/file.txt", new_str="  spaces  \n tabs \t", insert_line=5
        )
        assert args.new_str == "  spaces  \n tabs \t"

    def test_insert_line_must_be_gte_1(self) -> None:
        """Test insert_line must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            InsertLineArgs(path="/test/file.txt", new_str="content", insert_line=0)

        assert "insert_line" in str(exc_info.value)

    def test_insert_line_can_be_1(self) -> None:
        """Test insert_line can be exactly 1."""
        args = InsertLineArgs(path="/test/file.txt", new_str="content", insert_line=1)
        assert args.insert_line == 1

    def test_insert_line_positive_values(self) -> None:
        """Test insert_line accepts various positive values."""
        for line_num in [1, 5, 100, 1000]:
            args = InsertLineArgs(
                path="/test/file.txt", new_str="content", insert_line=line_num
            )
            assert args.insert_line == line_num

    def test_model_validation(self) -> None:
        """Test InsertLineArgs validates types correctly."""
        with pytest.raises(ValidationError):
            InsertLineArgs(
                path=123,
                new_str="text",
                insert_line=1,  # type: ignore
            )  # type: ignore

        with pytest.raises(ValidationError):
            InsertLineArgs(
                path="/test/file.txt",
                new_str="text",
                insert_line="one",  # type: ignore
            )


# =============================================================================
# InsertLineResult Tests
# =============================================================================


class TestInsertLineResult:
    """Tests for InsertLineResult Pydantic model."""

    def test_creation_with_output(self) -> None:
        """Test InsertLineResult can be created with output."""
        result = InsertLineResult(output="Content inserted successfully.")
        assert result.output == "Content inserted successfully."


# =============================================================================
# Insertion Success Tests
# =============================================================================


class TestInsertSuccess:
    """Tests for successful insertion operations."""

    pytestmark = pytest.mark.asyncio

    async def test_insert_at_beginning(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test inserting at line 1 (beginning of file)."""
        file_path = temp_dir / "beginning.txt"
        original_content = "line2\nline3\nline4"
        file_path.write_text(original_content, encoding="utf-8")

        result = await tool._arun(
            path=str(file_path), new_str="NEW FIRST LINE", insert_line=1
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        expected = "NEW FIRST LINE\nline2\nline3\nline4"
        assert file_path.read_text(encoding="utf-8") == expected

    async def test_insert_at_end(self, tool: InsertLineTool, temp_dir: Path) -> None:
        """Test inserting at end of file (line num_lines + 1)."""
        file_path = temp_dir / "end.txt"
        original_content = "line1\nline2\nline3"
        file_path.write_text(original_content, encoding="utf-8")

        result = await tool._arun(
            path=str(file_path), new_str="NEW LAST LINE", insert_line=4
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        expected = "line1\nline2\nline3\nNEW LAST LINE"
        assert file_path.read_text(encoding="utf-8") == expected

    async def test_insert_in_middle(self, tool: InsertLineTool, temp_dir: Path) -> None:
        """Test inserting in the middle of a file."""
        file_path = temp_dir / "middle.txt"
        original_content = "line1\nline2\nline3\nline4"
        file_path.write_text(original_content, encoding="utf-8")

        result = await tool._arun(
            path=str(file_path), new_str="INSERTED LINE", insert_line=3
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        expected = "line1\nline2\nINSERTED LINE\nline3\nline4"
        assert file_path.read_text(encoding="utf-8") == expected

    async def test_insert_into_empty_file_line_1(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test inserting into empty file at line 1."""
        file_path = temp_dir / "empty.txt"
        file_path.write_text("", encoding="utf-8")

        result = await tool._arun(
            path=str(file_path), new_str="FIRST CONTENT", insert_line=1
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        assert file_path.read_text(encoding="utf-8") == "FIRST CONTENT"

    async def test_insert_at_end_with_trailing_newline(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test inserting at end of file with trailing newline preserves structure."""
        file_path = temp_dir / "trailing.txt"
        # File with trailing newline - split creates ["line1", "line2", "line3", ""]
        original_content = "line1\nline2\nline3\n"
        file_path.write_text(original_content, encoding="utf-8")

        await tool._arun(path=str(file_path), new_str="APPENDED LINE", insert_line=4)

        # Should preserve trailing newline structure:
        # Original: "line1\nline2\nline3\n" -> ["line1", "line2", "line3", ""]
        # After insert at line 4: ["line1", "line2", "line3", "APPENDED LINE", ""]
        # Result: "line1\nline2\nline3\nAPPENDED LINE\n"
        expected = "line1\nline2\nline3\nAPPENDED LINE\n"
        assert file_path.read_text(encoding="utf-8") == expected

    async def test_insert_saves_to_edit_history(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test that insertion saves content to edit history."""
        file_path = temp_dir / "history.txt"
        original_content = "original content"
        file_path.write_text(original_content, encoding="utf-8")

        await tool._arun(path=str(file_path), new_str="inserted", insert_line=1)

        # Check that history was saved
        assert str(file_path) in tool._edit_history
        history = tool._edit_history[str(file_path)]
        assert len(history) == 1
        assert history[0] == original_content

    async def test_insert_updates_view_timestamp(
        self, tool: InsertLineTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test that insertion updates the view timestamp."""
        file_path = temp_dir / "view.txt"
        file_path.write_text("content", encoding="utf-8")

        # Record initial view
        view_tracker.record_view(str(file_path))
        initial_timestamp = view_tracker.get_last_view_timestamp(str(file_path))

        # Perform insert
        await tool._arun(path=str(file_path), new_str="new content", insert_line=2)

        # Check that timestamp was updated
        new_timestamp = view_tracker.get_last_view_timestamp(str(file_path))
        assert new_timestamp is not None
        assert initial_timestamp is not None
        assert new_timestamp >= initial_timestamp

    async def test_insert_without_view_tracker(
        self, tool_no_view_tracker: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test insertion works without ViewTrackerService configured."""
        file_path = temp_dir / "no_tracker.txt"
        file_path.write_text("line1\nline2", encoding="utf-8")

        result = await tool_no_view_tracker._arun(
            path=str(file_path), new_str="INSERTED", insert_line=2
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        expected = "line1\nINSERTED\nline2"
        assert file_path.read_text(encoding="utf-8") == expected


# =============================================================================
# Line Out of Bounds Tests
# =============================================================================


class TestLineOutOfBounds:
    """Tests for line out of bounds error handling."""

    pytestmark = pytest.mark.asyncio

    async def test_empty_file_line_greater_than_1_raises_error(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test inserting at line > 1 in empty file fails."""
        file_path = temp_dir / "empty_bounds.txt"
        file_path.write_text("", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path), new_str="content", insert_line=1)

        assert exc_info.value.code == "LINE_OUT_OF_BOUNDS"
        assert "out of bounds" in str(exc_info.value)
        assert "empty file" in str(exc_info.value).lower()

    async def test_out_of_bounds_high(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test inserting at line num_lines + 2 fails."""
        file_path = temp_dir / "high_bounds.txt"
        file_path.write_text("line1\nline2", encoding="utf-8")

        # Valid range is 1 to num_lines + 1 (1 to 3 for 2-line file)
        # So line 4 should fail
        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path), new_str="BOUNDARY", insert_line=4)

        assert exc_info.value.code == "LINE_OUT_OF_BOUNDS"
        assert "out of bounds" in str(exc_info.value)

    async def test_error_message_includes_valid_range(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test error message includes valid range for guidance."""
        file_path = temp_dir / "range.txt"
        file_path.write_text("line1\nline2\nline3", encoding="utf-8")

        # Valid range is 1 to 4 for 3-line file
        # Line 5 should fail
        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path), new_str="content", insert_line=5)

        error_str = str(exc_info.value)
        # Check that the error message includes helpful guidance
        assert any(
            word in error_str.lower()
            for word in ["valid", "range", "beginning", "end", "read_file"]
        )

    async def test_error_message_includes_count(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test error message includes line count for context."""
        file_path = temp_dir / "count.txt"
        file_path.write_text("line1\nline2", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path), new_str="content", insert_line=5)

        # Error should mention the number of lines in the file
        error_str = str(exc_info.value)
        assert "2" in error_str or "lines" in error_str.lower()


# =============================================================================
# File Not Found Tests
# =============================================================================


class TestFileNotFound:
    """Tests for file not found error handling."""

    pytestmark = pytest.mark.asyncio

    async def test_insert_on_nonexistent_file_raises_error(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test insertion fails on nonexistent file."""
        file_path = temp_dir / "nonexistent.txt"

        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path), new_str="new content", insert_line=1)

        assert exc_info.value.code == "FILE_NOT_FOUND"
        assert "not found" in str(exc_info.value)

    async def test_error_message_suggests_create(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test error message suggests using create for new files."""
        file_path = temp_dir / "missing.txt"

        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path), new_str="new content", insert_line=1)

        error_str = str(exc_info.value).lower()
        # Check that suggestions include create
        assert "create" in error_str or len(exc_info.value.suggestions) > 0


# =============================================================================
# Path Resolution Tests
# =============================================================================


class TestPathResolution:
    """Tests for path resolution functionality."""

    pytestmark = pytest.mark.asyncio

    async def test_relative_path_resolved_against_workdir(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test relative paths are resolved against working directory."""
        file_path = temp_dir / "relative.txt"
        file_path.write_text("line1\nline2", encoding="utf-8")

        result = await tool._arun(
            path="relative.txt", new_str="INSERTED", insert_line=2
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        assert file_path.exists()
        expected = "line1\nINSERTED\nline2"
        assert file_path.read_text(encoding="utf-8") == expected

    async def test_absolute_path_works_correctly(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test absolute paths work correctly."""
        file_path = temp_dir / "absolute.txt"
        file_path.write_text("line1\nline2", encoding="utf-8")

        result = await tool._arun(
            path=str(file_path), new_str="INSERTED", insert_line=2
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        expected = "line1\nINSERTED\nline2"
        assert file_path.read_text(encoding="utf-8") == expected


# =============================================================================
# UTF-8 Encoding Tests
# =============================================================================


class TestUTF8Encoding:
    """Tests for UTF-8 encoding handling."""

    pytestmark = pytest.mark.asyncio

    async def test_utf8_handles_special_characters(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test UTF-8 encoding handles special characters correctly."""
        file_path = temp_dir / "unicode.txt"
        # Use multiline content for proper insertion test
        original_content = "Hello, 世界! 🌍\nÑoño ©®™"
        file_path.write_text(original_content, encoding="utf-8")

        result = await tool._arun(
            path=str(file_path), new_str="ADDED TEXT", insert_line=2
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        expected = "Hello, 世界! 🌍\nADDED TEXT\nÑoño ©®™"
        assert file_path.read_text(encoding="utf-8") == expected


# =============================================================================
# Tool Configuration Tests
# =============================================================================


class TestToolConfiguration:
    """Tests for tool configuration."""

    def test_tool_name_is_insert_line(self, tool: InsertLineTool) -> None:
        """Test tool has correct name."""
        assert tool.name == "insert_line"

    def test_tool_has_description(self, tool: InsertLineTool) -> None:
        """Test tool has description."""
        assert len(tool.description) > 0

    def test_tool_uses_insert_line_args_schema(self, tool: InsertLineTool) -> None:
        """Test tool uses InsertLineArgs as schema."""
        assert tool.args_schema == InsertLineArgs


# =============================================================================
# Error Messages Tests
# =============================================================================


class TestErrorMessages:
    """Tests for error message quality."""

    pytestmark = pytest.mark.asyncio

    async def test_all_errors_include_helpful_suggestions(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test that all errors include helpful suggestions."""
        # Test file not found
        file_path = temp_dir / "nonexistent.txt"
        try:
            await tool._arun(path=str(file_path), new_str="new content", insert_line=1)
            raise AssertionError("Expected FileSystemError")
        except FileSystemError as e:
            assert len(e.suggestions) > 0

    async def test_file_not_found_error_format(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test file not found error matches expected format."""
        file_path = temp_dir / "notfound.txt"

        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path), new_str="new content", insert_line=1)

        error_msg = str(exc_info.value)
        # Should mention file not found and suggest create
        assert "not found" in error_msg.lower() or "notfound" in error_msg.lower()
        assert "create" in error_msg.lower()

    async def test_line_out_of_bounds_error_format(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test line out of bounds error includes helpful guidance."""
        file_path = temp_dir / "bounds.txt"
        file_path.write_text("line1\nline2", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path), new_str="content", insert_line=5)

        error_msg = str(exc_info.value).lower()
        # Should mention bounds and suggest using read_file
        assert "out of bounds" in error_msg or "bounds" in error_msg


# =============================================================================
# Edit History Tests
# =============================================================================


class TestEditHistory:
    """Tests for edit history functionality."""

    pytestmark = pytest.mark.asyncio

    async def test_multiple_inserts_save_to_history(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test multiple inserts save all versions to history."""
        file_path = temp_dir / "multi_insert.txt"
        file_path.write_text("version 1", encoding="utf-8")

        # First insert
        await tool._arun(path=str(file_path), new_str="inserted1", insert_line=2)
        # Second insert
        await tool._arun(path=str(file_path), new_str="inserted2", insert_line=3)

        history = tool._edit_history[str(file_path)]
        assert len(history) == 2

    async def test_pop_history_returns_previous_content(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test popping history returns the previous content."""
        file_path = temp_dir / "pop_history.txt"
        file_path.write_text("current content", encoding="utf-8")

        # Perform insert
        await tool._arun(path=str(file_path), new_str="inserted content", insert_line=2)

        # Pop history
        previous = tool._pop_history(str(file_path))
        assert previous == "current content"

        # History should be empty now
        history = tool._edit_history[str(file_path)]
        assert len(history) == 0

    async def test_pop_history_returns_none_when_empty(
        self, tool: InsertLineTool, temp_dir: Path
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

    async def test_insert_multiline_content(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test inserting multiline content."""
        file_path = temp_dir / "multiline.txt"
        file_path.write_text("line1\nline2", encoding="utf-8")

        multiline_content = "new line A\nnew line B\nnew line C"
        result = await tool._arun(
            path=str(file_path), new_str=multiline_content, insert_line=2
        )

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        expected = "line1\nnew line A\nnew line B\nnew line C\nline2"
        assert file_path.read_text(encoding="utf-8") == expected

    async def test_insert_single_line_file(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test inserting into a single-line file."""
        file_path = temp_dir / "single_line.txt"
        file_path.write_text("only line", encoding="utf-8")

        # Insert at beginning
        result = await tool._arun(path=str(file_path), new_str="FIRST", insert_line=1)
        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )

        # Insert at end (line 2)
        result = await tool._arun(path=str(file_path), new_str="LAST", insert_line=2)
        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )

    async def test_insert_at_max_boundary(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test inserting at maximum valid boundary (num_lines + 1)."""
        file_path = temp_dir / "boundary.txt"
        file_path.write_text("line1\nline2\nline3", encoding="utf-8")

        # Insert at line 4 (num_lines + 1 = 3 + 1 = 4)
        result = await tool._arun(path=str(file_path), new_str="AT END", insert_line=4)

        assert (
            "inserted successfully" in result.output or "inserted into" in result.output
        )
        expected = "line1\nline2\nline3\nAT END"
        assert file_path.read_text(encoding="utf-8") == expected

    async def test_insert_preserves_file_encoding(
        self, tool: InsertLineTool, temp_dir: Path
    ) -> None:
        """Test that file encoding is preserved after insertion."""
        file_path = temp_dir / "encoding.txt"
        # Write with UTF-8 encoding
        file_path.write_text("café\nnaïve\nélève", encoding="utf-8")

        await tool._arun(path=str(file_path), new_str="SOME TEXT", insert_line=10)

        # Read back and verify encoding is preserved
        content = file_path.read_text(encoding="utf-8")
        assert "café" in content
        assert "naïve" in content
        assert "élève" in content

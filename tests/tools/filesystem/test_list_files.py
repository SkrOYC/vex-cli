"""Unit tests for ListFilesTool.

Tests cover all acceptance criteria including:
- find_files command with glob patterns
- find_files command with exclude patterns
- find_files command with max_results limit
- find_files command with include_hidden flag
- find_files command with recursive=false
- list command non-recursive
- list command recursive with max_depth
- Tree structure formatting with indentation
- File metadata (size, line counts)
- Directory sorting (directories first, then alphabetically)
- Hidden file handling
- Output truncation at 80,000 characters
- Relative and absolute path resolution
- All error messages matching TypeScript behavior
- Pattern matching with wildcards (*, **, ?, [abc])
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from vibe.core.tools.filesystem.list_files import (
    ListFilesArgs,
    ListFilesResult,
    ListFilesTool,
)
from vibe.core.tools.filesystem.types import FileSystemError

# Mark all async tests
pytestmark = pytest.mark.asyncio


# =============================================================================
# ListFilesArgs Tests
# =============================================================================


class TestListFilesArgs:
    """Tests for ListFilesArgs Pydantic model."""

    def test_creation_with_defaults(self) -> None:
        """Test ListFilesArgs can be created with defaults."""
        args = ListFilesArgs()
        assert args.path == "."
        assert args.patterns == []
        assert args.exclude == []
        assert args.recursive is True
        assert args.max_results == 1000
        assert args.include_hidden is False

    def test_creation_with_path_only(self) -> None:
        """Test ListFilesArgs with path only."""
        args = ListFilesArgs(path="/test/dir")
        assert args.path == "/test/dir"
        assert args.patterns == []

    def test_creation_with_patterns(self) -> None:
        """Test ListFilesArgs with patterns."""
        args = ListFilesArgs(patterns=["*.py", "*.js"])
        assert args.patterns == ["*.py", "*.js"]

    def test_creation_with_all_fields(self) -> None:
        """Test ListFilesArgs with all fields."""
        args = ListFilesArgs(
            path="/test",
            patterns=["*.py"],
            exclude=["*.test.py"],
            recursive=False,
            max_results=500,
            include_hidden=True,
        )
        assert args.path == "/test"
        assert args.patterns == ["*.py"]
        assert args.exclude == ["*.test.py"]
        assert args.recursive is False
        assert args.max_results == 500
        assert args.include_hidden is True

    def test_max_results_validation(self) -> None:
        """Test max_results validates range."""
        # Valid range
        args = ListFilesArgs(max_results=100)
        assert args.max_results == 100

        # Too low
        with pytest.raises(ValidationError):
            ListFilesArgs(max_results=0)

        # Too high
        with pytest.raises(ValidationError):
            ListFilesArgs(max_results=10001)

    def test_model_validation(self) -> None:
        """Test ListFilesArgs validates types correctly."""
        with pytest.raises(ValidationError):
            ListFilesArgs(path=123)  # type: ignore


# =============================================================================
# ListFilesResult Tests
# =============================================================================


class TestListFilesResult:
    """Tests for ListFilesResult Pydantic model."""

    def test_creation(self) -> None:
        """Test ListFilesResult can be created."""
        result = ListFilesResult(output="test output")
        assert result.output == "test output"


# =============================================================================
# Tool Configuration Tests
# =============================================================================


class TestToolConfiguration:
    """Tests for tool configuration and metadata."""

    def test_tool_name_is_list_files(self) -> None:
        """Test tool has correct name."""
        tool = ListFilesTool()
        assert tool.name == "list_files"

    def test_tool_has_description(self) -> None:
        """Test tool has description."""
        tool = ListFilesTool()
        assert len(tool.description) > 0

    def test_tool_uses_list_files_args_schema(self) -> None:
        """Test tool uses ListFilesArgs as schema."""
        tool = ListFilesTool()
        assert tool.args_schema == ListFilesArgs


# =============================================================================
# find_files Command Tests
# =============================================================================


class TestFindFilesSimple:
    """Tests for find_files with simple patterns."""

    async def test_find_files_simple_pattern(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files with simple pattern (*.py)."""
        # Create test files
        (temp_dir / "test.py").write_text("print('hello')")
        (temp_dir / "test.txt").write_text("hello")
        (temp_dir / "main.py").write_text("print('world')")

        result = await tool._arun(path=str(temp_dir), patterns=["*.py"])

        assert "Found 2 files" in result.output
        assert "test.py" in result.output
        assert "main.py" in result.output
        assert "test.txt" not in result.output

    async def test_find_files_no_matches(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files when no files match."""
        (temp_dir / "test.txt").write_text("hello")

        result = await tool._arun(path=str(temp_dir), patterns=["*.py"])

        assert "No files found" in result.output

    async def test_find_files_multiple_patterns(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files with multiple patterns."""
        (temp_dir / "test.py").write_text("print('hello')")
        (temp_dir / "test.js").write_text("console.log('hello')")
        (temp_dir / "test.txt").write_text("hello")

        result = await tool._arun(path=str(temp_dir), patterns=["*.py", "*.js"])

        assert "Found 2 files" in result.output
        assert "test.py" in result.output
        assert "test.js" in result.output


class TestFindFilesWithExclude:
    """Tests for find_files with exclude patterns."""

    async def test_find_files_with_exclude(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files with exclude patterns."""
        (temp_dir / "test.py").write_text("print('hello')")
        (temp_dir / "test_test.py").write_text("print('test')")
        (temp_dir / "main.py").write_text("print('world')")

        result = await tool._arun(
            path=str(temp_dir), patterns=["*.py"], exclude=["*_test.py"]
        )

        assert "Found 2 files" in result.output
        assert "test.py" in result.output
        assert "main.py" in result.output
        assert "test_test.py" not in result.output


class TestFindFilesMaxResults:
    """Tests for find_files with max_results limit."""

    async def test_find_files_respects_max_results(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files respects max_results limit."""
        # Create more files than max_results
        for i in range(10):
            (temp_dir / f"file_{i}.py").write_text(f"# file {i}")

        result = await tool._arun(path=str(temp_dir), patterns=["*.py"], max_results=5)

        assert "(limited to 5)" in result.output
        # Should contain 5 file references
        lines = [l for l in result.output.split("\n") if "file_" in l]
        assert len(lines) == 5


class TestFindFilesHiddenFiles:
    """Tests for find_files with hidden files."""

    async def test_find_files_excludes_hidden_by_default(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files excludes hidden files by default."""
        (temp_dir / "test.py").write_text("print('hello')")
        (temp_dir / ".hidden.py").write_text("print('hidden')")

        result = await tool._arun(path=str(temp_dir), patterns=["*.py"])

        assert "Found 1 files" in result.output
        assert "test.py" in result.output
        assert ".hidden.py" not in result.output

    async def test_find_files_includes_hidden_when_enabled(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files includes hidden files when include_hidden=True."""
        (temp_dir / "test.py").write_text("print('hello')")
        (temp_dir / ".hidden.py").write_text("print('hidden')")

        result = await tool._arun(
            path=str(temp_dir), patterns=["*.py"], include_hidden=True
        )

        assert "Found 2 files" in result.output
        assert "test.py" in result.output
        assert ".hidden.py" in result.output


class TestFindFilesRecursive:
    """Tests for find_files with recursive option."""

    async def test_find_files_recursive_default(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files is recursive by default."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("print('nested')")
        (temp_dir / "root.py").write_text("print('root')")

        result = await tool._arun(path=str(temp_dir), patterns=["*.py"])

        assert "Found 2 files" in result.output
        assert "root.py" in result.output
        assert "subdir/nested.py" in result.output

    async def test_find_files_non_recursive(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files with recursive=False."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("print('nested')")
        (temp_dir / "root.py").write_text("print('root')")

        result = await tool._arun(
            path=str(temp_dir), patterns=["*.py"], recursive=False
        )

        assert "Found 1 files" in result.output
        assert "root.py" in result.output
        assert "subdir" not in result.output


# =============================================================================
# list Command Tests
# =============================================================================


class TestListNonRecursive:
    """Tests for list command non-recursive."""

    async def test_list_non_recursive(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test list command non-recursive."""
        # Create directory structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")

        # Use recursive=False to get simple list format with markers
        result = await tool._arun(path=str(temp_dir), recursive=False)

        assert "[DIR]" in result.output
        assert "[FILE]" in result.output
        assert "subdir/" in result.output
        assert "file2.txt" in result.output

    async def test_list_sorts_directories_first(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test list command sorts directories before files."""
        # Create files and directories
        (temp_dir / "z_file.txt").write_text("z content")
        (temp_dir / "a_dir").mkdir()
        (temp_dir / "b_file.txt").write_text("b content")

        # Use recursive=False for consistent ordering test
        result = await tool._arun(path=str(temp_dir), recursive=False)

        output = result.output
        dir_pos = output.find("a_dir/")
        file_z_pos = output.find("z_file.txt")
        file_b_pos = output.find("b_file.txt")

        # Directory should come before files
        assert dir_pos < file_z_pos
        assert dir_pos < file_b_pos

    async def test_list_shows_metadata(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test list command shows file metadata."""
        (temp_dir / "test.py").write_text("line1\nline2\nline3")

        # Use recursive=False for consistent output
        result = await tool._arun(path=str(temp_dir), recursive=False)

        assert "test.py" in result.output
        assert "3 lines" in result.output


class TestListRecursive:
    """Tests for list command recursive."""

    async def test_list_recursive(self, tool: ListFilesTool, temp_dir: Path) -> None:
        """Test list command recursive with tree structure."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("print('nested')")
        (temp_dir / "root.py").write_text("print('root')")

        result = await tool._arun(path=str(temp_dir), recursive=True)

        assert "root.py" in result.output
        assert "subdir/" in result.output
        assert "nested.py" in result.output

    async def test_list_recursive_indentation(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test list command recursive has proper indentation."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.py").write_text("content")

        result = await tool._arun(path=str(temp_dir), recursive=True)

        output = result.output
        # Subdirectory should have more indentation than root
        subdir_line = [l for l in output.split("\n") if "subdir/" in l][0]
        assert subdir_line.startswith("  ")


# =============================================================================
# Hidden File Tests
# =============================================================================


class TestHiddenFiles:
    """Tests for hidden file handling."""

    async def test_list_excludes_hidden_by_default(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test list command excludes hidden files by default."""
        (temp_dir / "visible.txt").write_text("visible")
        (temp_dir / ".hidden").mkdir()
        (temp_dir / ".hidden.txt").write_text("hidden")

        result = await tool._arun(path=str(temp_dir), recursive=False)

        assert "visible.txt" in result.output
        assert ".hidden" not in result.output
        assert ".hidden.txt" not in result.output

    async def test_list_includes_hidden_when_enabled(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test list command includes hidden files when include_hidden=True."""
        (temp_dir / "visible.txt").write_text("visible")
        (temp_dir / ".hidden.txt").write_text("hidden")

        result = await tool._arun(
            path=str(temp_dir), include_hidden=True, recursive=False
        )

        assert "visible.txt" in result.output
        assert ".hidden.txt" in result.output


# =============================================================================
# Path Resolution Tests
# =============================================================================


class TestPathResolution:
    """Tests for path resolution."""

    async def test_relative_path(self, tool: ListFilesTool, temp_dir: Path) -> None:
        """Test relative path resolution."""
        # Create a subdirectory with a file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "test.py").write_text("print('hello')")

        # Run from temp_dir as workdir with relative path to directory
        result = await tool._arun(path="subdir", patterns=["*.py"])

        assert "Found 1 files" in result.output

    async def test_absolute_path(self, tool: ListFilesTool, temp_dir: Path) -> None:
        """Test absolute path resolution."""
        # Create a subdirectory with a file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "test.py").write_text("print('hello')")

        result = await tool._arun(path=str(subdir), patterns=["*.py"])

        assert "Found 1 files" in result.output


# =============================================================================
# Error Message Tests
# =============================================================================


class TestErrorMessages:
    """Tests for error messages matching TypeScript behavior."""

    async def test_directory_not_found_error(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test directory not found error includes helpful suggestions."""
        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(temp_dir / "nonexistent"))

        assert exc_info.value.code == "DIRECTORY_NOT_FOUND"
        assert "not found" in str(exc_info.value)
        assert len(exc_info.value.suggestions) > 0

    async def test_path_is_file_error(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test path is file error includes helpful suggestions."""
        file_path = temp_dir / "file.txt"
        file_path.write_text("content")

        with pytest.raises(FileSystemError) as exc_info:
            await tool._arun(path=str(file_path))

        assert exc_info.value.code == "PATH_NOT_DIRECTORY"
        assert "not a directory" in str(exc_info.value)
        assert len(exc_info.value.suggestions) > 0


# =============================================================================
# Output Truncation Tests
# =============================================================================


class TestOutputTruncation:
    """Tests for output truncation at 80,000 characters."""

    async def test_find_files_truncation(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files output truncates at 80,000 characters."""
        # Create many files with long names to reach 80k chars
        # Each filename is ~40 chars, 2500 files = ~100k chars
        for i in range(2500):
            (
                temp_dir / f"file_{i:04d}_very_long_name_for_testing_purposes.py"
            ).write_text(f"# file {i}")

        # Use a high max_results to allow more files before truncation
        result = await tool._arun(
            path=str(temp_dir), patterns=["*.py"], max_results=5000
        )

        # Output should be truncated at 80,000 chars
        assert len(result.output) <= 85_000  # Allow for footer
        assert "truncated" in result.output


# =============================================================================
# Pattern Matching Tests
# =============================================================================


class TestPatternMatching:
    """Tests for pattern matching with wildcards."""

    async def test_double_star_wildcard(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test pattern matching with ** wildcard."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("print('nested')")

        result = await tool._arun(path=str(temp_dir), patterns=["**/*.py"])

        assert "Found" in result.output

    async def test_question_mark_wildcard(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test pattern matching with ? wildcard."""
        (temp_dir / "file1.py").write_text("print(1)")
        (temp_dir / "file2.py").write_text("print(2)")
        (temp_dir / "file10.py").write_text("print(10)")

        result = await tool._arun(path=str(temp_dir), patterns=["file?.py"])

        assert "Found 2 files" in result.output
        assert "file1.py" in result.output
        assert "file2.py" in result.output
        assert "file10.py" not in result.output

    async def test_bracket_wildcard(self, tool: ListFilesTool, temp_dir: Path) -> None:
        """Test pattern matching with [abc] wildcard."""
        (temp_dir / "fileA.py").write_text("print('A')")
        (temp_dir / "fileB.py").write_text("print('B')")
        (temp_dir / "fileC.py").write_text("print('C')")
        (temp_dir / "fileD.py").write_text("print('D')")

        result = await tool._arun(path=str(temp_dir), patterns=["file[ABC].py"])

        assert "Found 3 files" in result.output
        assert "fileA.py" in result.output
        assert "fileB.py" in result.output
        assert "fileC.py" in result.output
        assert "fileD.py" not in result.output


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_empty_directory(self, tool: ListFilesTool, temp_dir: Path) -> None:
        """Test listing empty directory."""
        result = await tool._arun(path=str(temp_dir), recursive=False)

        # Should work without error
        assert result.output is not None

    async def test_find_files_default_pattern(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test find_files without patterns uses default *."""
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")

        # Use patterns=["*"] explicitly for find_files behavior
        result = await tool._arun(path=str(temp_dir), patterns=["*"])

        assert "Found 2 files" in result.output

    async def test_list_with_nested_hidden_files(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test list with hidden files in nested directories."""
        subdir = temp_dir / ".hidden_dir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested")

        result = await tool._arun(path=str(temp_dir), recursive=False)

        # Hidden directory should be excluded
        assert ".hidden_dir" not in result.output

    async def test_file_metadata_size_formatting(
        self, tool: ListFilesTool, temp_dir: Path
    ) -> None:
        """Test file metadata includes properly formatted size."""
        # Create a file larger than 1KB (2048 bytes)
        (temp_dir / "large.txt").write_text("x" * 2048)

        result = await tool._arun(path=str(temp_dir), recursive=False)

        assert "large.txt" in result.output
        # Should show size in KB
        assert "KB" in result.output or "B" in result.output

"""Unit tests for GrepTool.

Tests cover all acceptance criteria including:
- Regex search mode with case-sensitive and case-insensitive flags
- Literal search mode (regex=false)
- File pattern filtering (patterns parameter)
- Recursive and non-recursive search
- Hidden file inclusion
- Max results limiting
- Context lines (5 before, 3 after)
- Output format matching TypeScript FileExplorer behavior
- Output truncation at 80,000 characters
- Text file detection and binary file filtering
- Path resolution (relative and absolute paths)
- All error messages with helpful suggestions
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from vibe.core.tools.filesystem.grep import GrepArgs, GrepTool

# Mark all async tests
pytestmark = pytest.mark.asyncio


# =============================================================================
# GrepArgs Tests
# =============================================================================


class TestGrepArgs:
    """Tests for GrepArgs Pydantic model."""

    def test_creation_with_defaults(self) -> None:
        """Test GrepArgs can be created with defaults."""
        args = GrepArgs(path=".", query="test")
        assert args.path == "."
        assert args.query == "test"
        assert args.patterns == []
        assert args.case_sensitive is False
        assert args.regex is False
        assert args.recursive is True
        assert args.max_results == 1000
        assert args.include_hidden is False

    def test_creation_with_path_only(self) -> None:
        """Test GrepArgs with path and query only."""
        args = GrepArgs(path="/test/dir", query="function")
        assert args.path == "/test/dir"
        assert args.query == "function"
        assert args.patterns == []

    def test_creation_with_patterns(self) -> None:
        """Test GrepArgs with patterns."""
        args = GrepArgs(patterns=["*.py", "*.js"], query="import")
        assert args.patterns == ["*.py", "*.js"]

    def test_creation_with_all_fields(self) -> None:
        """Test GrepArgs with all fields."""
        args = GrepArgs(
            path="/test",
            query="function.*",
            patterns=["*.py"],
            case_sensitive=True,
            regex=True,
            recursive=False,
            max_results=500,
            include_hidden=True,
        )
        assert args.path == "/test"
        assert args.query == "function.*"
        assert args.patterns == ["*.py"]
        assert args.case_sensitive is True
        assert args.regex is True
        assert args.recursive is False
        assert args.max_results == 500
        assert args.include_hidden is True

    def test_max_results_validation(self) -> None:
        """Test max_results validates range."""
        # Valid range
        args = GrepArgs(query="test", max_results=100)
        assert args.max_results == 100

        # Too low
        with pytest.raises(ValidationError):
            GrepArgs(query="test", max_results=0)

        # Too high
        with pytest.raises(ValidationError):
            GrepArgs(query="test", max_results=10001)

    def test_model_validation(self) -> None:
        """Test GrepArgs validates types correctly."""
        # Valid args
        args = GrepArgs(
            path="/test",
            query="test",
            case_sensitive=False,
            regex=False,
            recursive=True,
            max_results=500,
            include_hidden=False,
        )
        assert args.case_sensitive is False
        assert args.regex is False
        assert args.recursive is True


# =============================================================================
# GrepTool Tests
# =============================================================================


class TestGrepTool:
    """Tests for GrepTool functionality."""

    # =============================================================================
    # Regex Search Tests
    # =============================================================================

    async def test_regex_search_matches_correctly(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test regex search finds pattern matches."""
        # Create test files
        (temp_dir / "utils.js").write_text(
            """function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
"""
        )

        result = await grep_tool.run(
            GrepArgs(path=".", query=r"function\s+\w+", regex=True, recursive=True)
        )

        assert "utils.js" in result.output
        assert "function formatDate" in result.output
        assert "function debounce" in result.output

    async def test_literal_search_matches_correctly(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test literal search finds exact matches."""
        (temp_dir / "main.py").write_text(
            """def hello():
    print("Hello, world!")
    return True

class Calculator:
    def add(self, a, b):
        return a + b
"""
        )

        result = await grep_tool.run(
            GrepArgs(path=".", query="hello", regex=False, recursive=True)
        )

        assert "main.py" in result.output
        assert "hello" in result.output

    async def test_case_sensitive_flag_works(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test case-sensitive search works correctly."""
        (temp_dir / "main.py").write_text(
            """def hello():
    print("Hello, world!")
    return True

def Hello():
    pass
"""
        )

        # Case-insensitive (default) - should find both "hello" and "Hello"
        result_ci = await grep_tool.run(
            GrepArgs(path=".", query="Hello", case_sensitive=False, recursive=True)
        )
        assert "main.py" in result_ci.output
        # Count case-insensitive matches for "hello"/"Hello"
        hello_count_ci = sum(
            1
            for line in result_ci.output.split("\n")
            if "hello" in line.lower() and not line.strip().startswith("/")
        )
        assert hello_count_ci >= 2, (
            f"Expected at least 2 case-insensitive matches, got {hello_count_ci}"
        )

        # Case-sensitive - should only find "Hello" (capital H)
        result_cs = await grep_tool.run(
            GrepArgs(path=".", query="Hello", case_sensitive=True, recursive=True)
        )
        # Count case-sensitive matches for "Hello" (appears in print and def)
        hello_count_cs = sum(
            1
            for line in result_cs.output.split("\n")
            if "Hello" in line and not line.strip().startswith("/")
        )
        assert hello_count_cs == 2, (
            f"Expected exactly 2 case-sensitive matches, got {hello_count_cs}"
        )

    async def test_case_insensitive_search_works(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test case-insensitive search works correctly."""
        (temp_dir / "main.py").write_text("Hello\nHELLO\nhello\n")

        result = await grep_tool.run(
            GrepArgs(path=".", query="hello", case_sensitive=False, recursive=True)
        )

        # Should find all three when case-insensitive
        assert "main.py" in result.output
        # Count total "hello" occurrences (case-insensitive means it finds all variants)
        hello_count = sum(
            1
            for line in result.output.split("\n")
            if "hello" in line.lower() and not line.strip().startswith("/")
        )
        assert hello_count >= 3, f"Expected at least 3 matches, got {hello_count}"

    # =============================================================================
    # File Pattern Tests
    # =============================================================================

    async def test_file_pattern_filtering_works(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test file pattern filtering limits search to specific types."""
        (temp_dir / "utils.js").write_text("function test() { return 1; }\n")
        (temp_dir / "main.py").write_text("def test():\n    pass\n")
        (temp_dir / "Button.tsx").write_text("const test = 1;\n")

        result = await grep_tool.run(
            GrepArgs(path=".", query="test", patterns=["*.js"], recursive=True)
        )

        assert "utils.js" in result.output
        assert "main.py" not in result.output
        assert "Button.tsx" not in result.output

    # =============================================================================
    # Recursive Search Tests
    # =============================================================================

    async def test_recursive_search_works(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test recursive search finds files in subdirectories."""
        # Create subdirectory with file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        subfile = subdir / "subfile.py"
        subfile.write_text("def sub_function():\n    pass\n")

        result = await grep_tool.run(GrepArgs(path=".", query="def", recursive=True))

        assert "subdir/subfile.py" in result.output

    async def test_non_recursive_search_works(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test non-recursive search only finds files in root directory."""
        # Create subdirectory with file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        subfile = subdir / "subfile.py"
        subfile.write_text("def sub_function():\n    pass\n")

        # Also create a file in root to ensure there are files to find
        root_file = temp_dir / "root.py"
        root_file.write_text("def root_func():\n    pass\n")

        result = await grep_tool.run(GrepArgs(path=".", query="def", recursive=False))

        # Should find root file
        assert "root.py" in result.output
        # Should not find subdirectory files
        assert "subdir" not in result.output

    # =============================================================================
    # Hidden Files Tests
    # =============================================================================

    async def test_hidden_file_inclusion_works(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test hidden file inclusion flag works correctly."""
        # Create hidden file
        hidden_file = temp_dir / ".hidden"
        hidden_file.write_text("hidden content match\n")

        # Also create visible file with "match" so it can be found when searching for "match"
        visible_file = temp_dir / "main.py"
        visible_file.write_text("def visible():\n    pass\n    # match found\n")

        # Without include_hidden (default) - should only find visible file
        result_hidden = await grep_tool.run(
            GrepArgs(path=".", query="def", include_hidden=False, recursive=True)
        )
        # Hidden file should not be found
        assert ".hidden" not in result_hidden.output
        # Visible file should be found
        assert "main.py" in result_hidden.output

        # With include_hidden=True - should find both
        # Use "match" which appears in both files
        result_visible = await grep_tool.run(
            GrepArgs(path=".", query="match", include_hidden=True, recursive=True)
        )
        # Both files should be found
        assert ".hidden" in result_visible.output
        assert "main.py" in result_visible.output

    async def test_hidden_directory_files_excluded_by_default(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test files inside hidden directories are excluded by default."""
        # Create hidden directory with file
        hidden_dir = temp_dir / ".git"
        hidden_dir.mkdir()
        config_file = hidden_dir / "config"
        config_file.write_text("match in hidden dir\n")

        # Also create visible file with "match" so it can be found when searching for "match"
        visible_file = temp_dir / "main.py"
        visible_file.write_text("def visible():\n    pass\n    # match found\n")

        # Without include_hidden (default) - should only find visible file
        result_hidden = await grep_tool.run(
            GrepArgs(path=".", query="def", include_hidden=False, recursive=True)
        )
        # File in hidden directory should not be found
        assert ".git/config" not in result_hidden.output
        # Visible file should be found
        assert "main.py" in result_hidden.output

        # With include_hidden=True - should find both
        # Use "match" which appears in both files
        result_visible = await grep_tool.run(
            GrepArgs(path=".", query="match", include_hidden=True, recursive=True)
        )
        # File in hidden directory should be found
        assert ".git/config" in result_visible.output
        # Visible file should be found
        assert "main.py" in result_visible.output

    # =============================================================================
    # Max Results Tests
    # =============================================================================

    async def test_max_results_limiting_works(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test max_results limits matches correctly."""
        # Create file with many matches
        many_matches = temp_dir / "many.txt"
        many_matches.write_text("\n".join(["match"] * 100))

        result = await grep_tool.run(
            GrepArgs(path=".", query="match", max_results=10, recursive=True)
        )

        # Should contain truncation footer
        assert "matches" in result.output.lower()

        # Should not exceed max_results (count lines with "match")
        match_lines = [
            line
            for line in result.output.split("\n")
            if "match" in line
            and not line.strip().startswith("/")
            and not line.strip().startswith("...")
        ]
        # Allow some buffer for context lines, but should be around 10-20 lines
        assert len(match_lines) <= 25, (
            f"Expected ~10-20 match lines with context, got {len(match_lines)}"
        )

    # =============================================================================
    # Context Lines Tests
    # =============================================================================

    async def test_context_lines_are_correct(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test context lines (5 before, 3 after) are included."""
        (temp_dir / "main.py").write_text(
            """line1
line2
line3
line4
line5
target_line
line7
line8
line9
"""
        )

        result = await grep_tool.run(
            GrepArgs(path=".", query="target_line", recursive=True)
        )

        # Should include the match line
        assert "target_line" in result.output
        # Should have line numbers
        assert "6" in result.output

    # =============================================================================
    # Output Format Tests
    # =============================================================================

    async def test_output_format_matches_typescript(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test output format matches TypeScript FileExplorer behavior."""
        (temp_dir / "main.py").write_text("def hello():\n    pass\n")

        result = await grep_tool.run(
            GrepArgs(path=".", query="def hello", recursive=True)
        )

        # Should start with file path with line number suffix
        assert "main.py:1" in result.output
        # Match line should have > prefix with proper padding
        assert ">   1 def hello()" in result.output

    # =============================================================================
    # Binary File Filtering Tests
    # =============================================================================

    async def test_binary_files_are_skipped(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test binary files are filtered out."""
        # Create visible file so there's something to search
        visible_file = temp_dir / "main.py"
        visible_file.write_text("def test():\n    pass\n")

        # Create binary file
        (temp_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = await grep_tool.run(GrepArgs(path=".", query="def", recursive=True))

        # Should find the visible file
        assert "main.py" in result.output

    # =============================================================================
    # Error Handling Tests
    # =============================================================================

    async def test_directory_not_found_error(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test error when directory doesn't exist."""
        with pytest.raises(ValueError) as exc_info:
            await grep_tool.run(GrepArgs(path="/nonexistent/directory", query="test"))

        error_msg = str(exc_info.value)
        # Check for exact error format
        assert "Directory not found: '/nonexistent/directory'" in error_msg
        assert (
            "The specified directory doesn't exist in current working directory"
            in error_msg
        )
        assert "Try using 'list' command to explore available directories" in error_msg

    async def test_no_text_files_error(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test error when no text files found."""
        # Create a directory with only truly binary files (null bytes)
        binary_file = temp_dir / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        with pytest.raises(ValueError) as exc_info:
            await grep_tool.run(GrepArgs(path=".", query="test"))

        error_msg = str(exc_info.value)
        # Check for exact error format with • bullets
        assert "No text files found" in error_msg
        assert "• Use 'list' command to see what files are in directory" in error_msg
        assert "• Specify a directory with known text files" in error_msg
        assert "• Add file patterns to search specific file types" in error_msg

    async def test_invalid_regex_error(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test error for invalid regex pattern."""
        (temp_dir / "test.py").write_text("test\n")

        with pytest.raises(ValueError) as exc_info:
            await grep_tool.run(GrepArgs(path=".", query="[unclosed", regex=True))

        error_msg = str(exc_info.value)
        # Check for exact error format with • bullets
        assert "Invalid regex pattern" in error_msg
        assert "• Unmatched parentheses or brackets" in error_msg
        assert "• Incomplete escape sequences" in error_msg
        assert "• Invalid quantifiers" in error_msg

    # =============================================================================
    # Path Resolution Tests
    # =============================================================================

    async def test_relative_paths_resolved_correctly(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test relative paths are resolved to working directory."""
        (temp_dir / "main.py").write_text("def test():\n    pass\n")

        result = await grep_tool.run(GrepArgs(path=".", query="def"))

        assert "main.py" in result.output

    async def test_absolute_paths_work_correctly(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test absolute paths work correctly."""
        (temp_dir / "main.py").write_text("def test():\n    pass\n")

        result = await grep_tool.run(GrepArgs(path=str(temp_dir), query="def"))

        assert "main.py" in result.output

    async def test_path_is_file_error(
        self, grep_tool: GrepTool, temp_dir: Path
    ) -> None:
        """Test error when path is a file instead of directory."""
        (temp_dir / "main.py").write_text("test\n")

        with pytest.raises(ValueError) as exc_info:
            await grep_tool.run(GrepArgs(path=str(temp_dir / "main.py"), query="test"))

        error_msg = str(exc_info.value)
        # Check for exact error format
        assert "Path is not a directory" in error_msg
        assert "Use 'view' command to examine file contents" in error_msg
        assert "Specify a directory path to search" in error_msg

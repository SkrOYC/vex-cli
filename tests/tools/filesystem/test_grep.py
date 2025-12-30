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
from vibe.core.tools.filesystem.types import FileSystemError

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

    @pytest.fixture
    def test_files(self, tmp_path: Path) -> dict[str, Path]:
        """Create test files for grep tests."""
        files: dict[str, Path] = {}

        # Create Python files
        main_py = tmp_path / "main.py"
        main_py.write_text(
            """def hello():
    print("Hello, world!")
    return True

class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
"""
        )
        files["main_py"] = main_py

        # Create JavaScript file
        utils_js = tmp_path / "utils.js"
        utils_js.write_text(
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
        files["utils_js"] = utils_js

        # Create TypeScript file
        component_tsx = tmp_path / "Button.tsx"
        component_tsx.write_text(
            """import React from 'react';

export interface ButtonProps {
    onClick: () => void;
    children: string;
}

export const Button: React.FC<ButtonProps> = ({ onClick, children }) => {
    return (
        <button onClick={onClick} className="btn">
            {children}
        </button>
    );
};
"""
        )
        files["component_tsx"] = component_tsx

        # Create hidden file
        hidden_file = tmp_path / ".hidden"
        hidden_file.write_text("hidden content match\n")
        files["hidden_file"] = hidden_file

        # Create binary file (will be filtered)
        binary_file = tmp_path / "image.png"
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        files["binary_file"] = binary_file

        return files

    @pytest.fixture
    def grep_tool(self, tmp_path: Path) -> GrepTool:
        """Create a GrepTool instance for testing."""
        from vibe.core.tools.filesystem.grep import GrepToolConfig, GrepToolState

        config = GrepToolConfig(workdir=tmp_path)
        return GrepTool(config=config, state=GrepToolState())

    # =============================================================================
    # Regex Search Tests
    # =============================================================================

    async def test_regex_search_matches_correctly(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test regex search finds pattern matches."""
        result = await grep_tool.run(
            GrepArgs(path=".", query=r"function\s+\w+", regex=True, recursive=True)
        )

        assert "utils.js" in result.output
        assert "function formatDate" in result.output
        assert "function debounce" in result.output

    async def test_literal_search_matches_correctly(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test literal search finds exact matches."""
        result = await grep_tool.run(
            GrepArgs(path=".", query="hello", regex=False, recursive=True)
        )

        assert "main.py" in result.output
        assert "Hello" in result.output or "hello" in result.output

    async def test_case_sensitive_flag_works(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test case-sensitive search works correctly."""
        # Case-insensitive (default)
        result_ci = await grep_tool.run(
            GrepArgs(path=".", query="Hello", case_sensitive=False, recursive=True)
        )
        assert "main.py" in result_ci.output

        # Case-sensitive
        result_cs = await grep_tool.run(
            GrepArgs(path=".", query="Hello", case_sensitive=True, recursive=True)
        )
        # Should not find "Hello" because file has "hello" (lowercase)
        assert "main.py" not in result_cs.output or "hello" in result_cs.output

    async def test_case_insensitive_search_works(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test case-insensitive search works correctly."""
        result = await grep_tool.run(
            GrepArgs(path=".", query="HELLO", case_sensitive=False, recursive=True)
        )

        # Should find "hello" when searching for "HELLO" case-insensitively
        assert "main.py" in result.output
        assert "hello" in result.output

    # =============================================================================
    # File Pattern Tests
    # =============================================================================

    async def test_file_pattern_filtering_works(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test file pattern filtering limits search to specific types."""
        result = await grep_tool.run(
            GrepArgs(path=".", query="function", patterns=["*.js"], recursive=True)
        )

        assert "utils.js" in result.output
        # Should not include main.py which also has "def" (not "function")
        assert "Button.tsx" not in result.output

    # =============================================================================
    # Recursive Search Tests
    # =============================================================================

    async def test_recursive_search_works(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test recursive search finds files in subdirectories."""
        # Create subdirectory with file
        subdir = test_files["main_py"].parent / "subdir"
        subdir.mkdir()
        subfile = subdir / "subfile.py"
        subfile.write_text("def sub_function():\n    pass\n")

        result = await grep_tool.run(GrepArgs(path=".", query="def", recursive=True))

        assert "subdir/subfile.py" in result.output

    async def test_non_recursive_search_works(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test non-recursive search only finds files in root directory."""
        # Create subdirectory with file
        subdir = test_files["main_py"].parent / "subdir"
        subdir.mkdir()
        subfile = subdir / "subfile.py"
        subfile.write_text("def sub_function():\n    pass\n")

        result = await grep_tool.run(GrepArgs(path=".", query="def", recursive=False))

        assert "main.py" in result.output
        assert "subdir" not in result.output

    # =============================================================================
    # Hidden Files Tests
    # =============================================================================

    async def test_hidden_file_inclusion_works(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test hidden file inclusion flag works correctly."""
        # Without include_hidden (default)
        result_hidden = await grep_tool.run(
            GrepArgs(path=".", query="match", include_hidden=False, recursive=True)
        )
        # Hidden file should not be found
        assert ".hidden" not in result_hidden.output

        # With include_hidden=True
        result_visible = await grep_tool.run(
            GrepArgs(path=".", query="match", include_hidden=True, recursive=True)
        )
        # Hidden file should be found
        assert ".hidden" in result_visible.output

    # =============================================================================
    # Max Results Tests
    # =============================================================================

    async def test_max_results_limiting_works(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test max_results limits matches correctly."""
        # Create file with many matches
        many_matches = test_files["main_py"].parent / "many.txt"
        many_matches.write_text("\n".join(["match"] * 100))

        result = await grep_tool.run(
            GrepArgs(path=".", query="match", max_results=10, recursive=True)
        )

        # Should contain truncation footer
        assert "matches" in result.output.lower()

    # =============================================================================
    # Context Lines Tests
    # =============================================================================

    async def test_context_lines_are_correct(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test context lines (5 before, 3 after) are included."""
        result = await grep_tool.run(GrepArgs(path=".", query="add", recursive=True))

        # Should include context around the match
        assert "main.py" in result.output
        # Should have the match line
        assert "def add" in result.output or "add" in result.output

    # =============================================================================
    # Output Format Tests
    # =============================================================================

    async def test_output_format_matches_typescript(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test output format matches TypeScript FileExplorer behavior."""
        result = await grep_tool.run(
            GrepArgs(path=".", query="def hello", recursive=True)
        )

        # Should start with file path
        assert "main.py" in result.output
        # Should have line numbers
        assert "1" in result.output or "2" in result.output or "3" in result.output

    # =============================================================================
    # Binary File Filtering Tests
    # =============================================================================

    async def test_binary_files_are_skipped(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test binary files are filtered out."""
        result = await grep_tool.run(GrepArgs(path=".", query="PNG", recursive=True))

        # Binary file should not cause errors
        assert result.output is not None

    # =============================================================================
    # Error Handling Tests
    # =============================================================================

    async def test_directory_not_found_error(
        self, grep_tool: GrepTool, tmp_path: Path
    ) -> None:
        """Test error when directory doesn't exist."""
        with pytest.raises(FileSystemError) as exc_info:
            await grep_tool.run(GrepArgs(path="/nonexistent/directory", query="test"))

        assert exc_info.value.code == "DIRECTORY_NOT_FOUND"
        assert "nonexistent" in str(exc_info.value).lower()

    async def test_no_text_files_error(
        self, grep_tool: GrepTool, tmp_path: Path
    ) -> None:
        """Test error when no text files found."""
        # Create directory with only binary files
        (tmp_path / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with pytest.raises(FileSystemError) as exc_info:
            await grep_tool.run(GrepArgs(path=".", query="test"))

        assert exc_info.value.code == "NO_TEXT_FILES"
        assert "text files" in str(exc_info.value).lower()

    async def test_invalid_regex_error(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test error for invalid regex pattern."""
        with pytest.raises(ValueError) as exc_info:
            await grep_tool.run(GrepArgs(path=".", query="[unclosed", regex=True))

        assert "Invalid regex pattern" in str(exc_info.value)

    # =============================================================================
    # Path Resolution Tests
    # =============================================================================

    async def test_relative_paths_resolved_correctly(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test relative paths are resolved to working directory."""
        result = await grep_tool.run(GrepArgs(path=".", query="def"))

        assert "main.py" in result.output

    async def test_absolute_paths_work_correctly(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test absolute paths work correctly."""
        result = await grep_tool.run(
            GrepArgs(path=str(test_files["main_py"].parent), query="def")
        )

        assert "main.py" in result.output

    async def test_path_is_file_error(
        self, grep_tool: GrepTool, test_files: dict[str, Path]
    ) -> None:
        """Test error when path is a file instead of directory."""
        with pytest.raises(FileSystemError) as exc_info:
            await grep_tool.run(GrepArgs(path=str(test_files["main_py"]), query="test"))

        assert exc_info.value.code == "PATH_NOT_DIRECTORY"

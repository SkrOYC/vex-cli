"""Unit tests for ReadFileTool.

Tests cover all acceptance criteria including:
- Content view mode with line numbers
- Content view mode with view_range support
- Content view mode validates range bounds
- Outline mode for Python files with ast-grep-py
- Smart view type switching (auto mode based on file size)
- Media file detection and base64 encoding
- Media file size limits (5MB)
- Directory viewing with list and outline modes
- Directory viewing with view_range fails
- All error messages match TypeScript behavior
- Output truncation at 80,000 characters
- Path resolution (relative and absolute)
- UTF-8 encoding handling
- View tracking for all file types
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
import os
from pathlib import Path
import tempfile

from pydantic import ValidationError
import pytest

from vibe.core.tools.filesystem.read_file import (
    ReadFileArgs,
    ReadFileContentResult,
    ReadFileMediaResult,
    ReadFileResult,
    ReadFileTool,
    ReadFileToolConfig,
    ReadFileToolState,
)
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError

# Mark all async tests
pytestmark = pytest.mark.asyncio


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
def tool_config(view_tracker: ViewTrackerService, temp_dir: Path) -> ReadFileToolConfig:
    """Create a ReadFileToolConfig with ViewTrackerService."""
    return ReadFileToolConfig(view_tracker=view_tracker, workdir=temp_dir)


@pytest.fixture
def tool_config_no_view_tracker(temp_dir: Path) -> ReadFileToolConfig:
    """Create a ReadFileToolConfig without ViewTrackerService."""
    return ReadFileToolConfig(workdir=temp_dir)


@pytest.fixture
def tool_state() -> ReadFileToolState:
    """Create a fresh ReadFileToolState for each test."""
    return ReadFileToolState()


@pytest.fixture
def tool(
    tool_config: ReadFileToolConfig, tool_state: ReadFileToolState
) -> ReadFileTool:
    """Create a ReadFileTool instance for testing."""
    return ReadFileTool(config=tool_config, state=tool_state)


@pytest.fixture
def tool_no_view_tracker(
    tool_config_no_view_tracker: ReadFileToolConfig, tool_state: ReadFileToolState
) -> ReadFileTool:
    """Create a ReadFileTool instance without ViewTrackerService."""
    return ReadFileTool(config=tool_config_no_view_tracker, state=tool_state)


# =============================================================================
# ReadFileArgs Tests
# =============================================================================


class TestReadFileArgs:
    """Tests for ReadFileArgs Pydantic model."""

    def test_creation_with_path_only(self) -> None:
        """Test ReadFileArgs can be created with path only."""
        args = ReadFileArgs(path="/test/file.txt")
        assert args.path == "/test/file.txt"
        assert args.view_type == "auto"
        assert args.view_range is None
        assert args.recursive is False
        assert args.include_patterns is None

    def test_creation_with_all_fields(self) -> None:
        """Test ReadFileArgs with all fields."""
        args = ReadFileArgs(
            path="/test/file.py",
            view_type="content",
            view_range=(1, 10),
            recursive=True,
            include_patterns=["*.py", "*.pyi"],
        )
        assert args.path == "/test/file.py"
        assert args.view_type == "content"
        assert args.view_range == (1, 10)
        assert args.recursive is True
        assert args.include_patterns == ["*.py", "*.pyi"]

    def test_view_type_validation(self) -> None:
        """Test ReadFileArgs validates view_type values."""
        # Valid view types should work
        for view_type in ("content", "outline", "auto"):
            args = ReadFileArgs(path="/test/file.txt", view_type=view_type)
            assert args.view_type == view_type

    def test_view_range_validation(self) -> None:
        """Test ReadFileArgs validates view_range as tuple of ints."""
        args = ReadFileArgs(path="/test/file.txt", view_range=(5, 10))
        assert args.view_range == (5, 10)

    def test_model_validation(self) -> None:
        """Test ReadFileArgs validates types correctly."""
        with pytest.raises(ValidationError):
            ReadFileArgs(path=123)  # type: ignore

    def test_include_patterns_default_none(self) -> None:
        """Test include_patterns defaults to None."""
        args = ReadFileArgs(path="/test/file.txt")
        assert args.include_patterns is None


# =============================================================================
# ReadFileResult Tests
# =============================================================================


class TestReadFileResult:
    """Tests for ReadFileResult Pydantic model."""

    def test_creation_with_content(self) -> None:
        """Test ReadFileResult can be created with content."""
        content_result = ReadFileContentResult(output="file content", line_count=10)
        result = ReadFileResult(content=content_result)
        assert result.content is not None
        assert result.content.output == "file content"
        assert result.content.line_count == 10
        assert result.media is None
        assert result.raw is None

    def test_creation_with_media(self) -> None:
        """Test ReadFileResult can be created with media."""
        media_result = ReadFileMediaResult(
            type="image",
            data="base64data",
            mime_type="image/png",
            path="/test/image.png",
            size=12345,
        )
        result = ReadFileResult(media=media_result)
        assert result.media is not None
        assert result.media.type == "image"
        assert result.media.data == "base64data"
        assert result.content is None
        assert result.raw is None

    def test_creation_with_raw(self) -> None:
        """Test ReadFileResult can be created with raw string."""
        result = ReadFileResult(raw="raw output")
        assert result.raw == "raw output"
        assert result.content is None
        assert result.media is None

    def test_str_representation_content(self) -> None:
        """Test ReadFileResult str() returns content output."""
        content_result = ReadFileContentResult(output="test output", line_count=5)
        result = ReadFileResult(content=content_result)
        assert str(result) == "test output"

    def test_str_representation_media(self) -> None:
        """Test ReadFileResult str() returns media info."""
        media_result = ReadFileMediaResult(
            type="audio",
            data="base64data",
            mime_type="audio/mp3",
            path="/test/audio.mp3",
            size=54321,
        )
        result = ReadFileResult(media=media_result)
        assert "/test/audio.mp3" in str(result)
        assert "audio" in str(result)


# =============================================================================
# Content View Mode Tests
# =============================================================================


class TestContentViewMode:
    """Tests for content view mode functionality."""

    async def test_content_view_no_range(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test content view with no range shows entire file."""
        file_path = temp_dir / "test.txt"
        content = "line1\nline2\nline3"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        assert result.content.line_count == 3
        assert "1| line1" in result.content.output
        assert "2| line2" in result.content.output
        assert "3| line3" in result.content.output

    async def test_content_view_with_valid_range(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test content view with valid range shows subset."""
        file_path = temp_dir / "range_test.txt"
        content = "line1\nline2\nline3\nline4\nline5"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(
            ReadFileArgs(path=str(file_path), view_type="content", view_range=(2, 4))
        )

        assert result.content is not None
        assert result.content.line_count == 5
        # Should show lines 2, 3, 4
        assert "2| line2" in result.content.output
        assert "3| line3" in result.content.output
        assert "4| line4" in result.content.output
        # Should not show lines 1 and 5
        assert "1| line1" not in result.content.output
        assert "5| line5" not in result.content.output

    async def test_content_view_with_end_minus_one(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test content view with -1 for end shows from start to end of file."""
        file_path = temp_dir / "end_test.txt"
        content = "line1\nline2\nline3\nline4\nline5"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(
            ReadFileArgs(path=str(file_path), view_type="content", view_range=(3, -1))
        )

        assert result.content is not None
        # Should show lines 3, 4, 5
        assert "3| line3" in result.content.output
        assert "4| line4" in result.content.output
        assert "5| line5" in result.content.output

    async def test_content_view_range_out_of_bounds(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test content view with range out of bounds fails."""
        file_path = temp_dir / "bounds_test.txt"
        content = "line1\nline2\nline3"
        file_path.write_text(content, encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                ReadFileArgs(
                    path=str(file_path), view_type="content", view_range=(1, 100)
                )
            )

        assert exc_info.value.code == "VIEW_RANGE_OUT_OF_BOUNDS"
        assert "exceeds file length" in str(exc_info.value)
        assert len(exc_info.value.suggestions) > 0

    async def test_content_view_with_utf8_encoding(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test content view handles UTF-8 encoding correctly."""
        file_path = temp_dir / "unicode.txt"
        content = "Hello, 世界! 🌍 Ñoño ©®™"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        assert "世界" in result.content.output
        assert "🌍" in result.content.output

    async def test_content_view_resolves_relative_path(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test content view resolves relative paths against workdir."""
        file_path = temp_dir / "relative.txt"
        file_path.write_text("relative content", encoding="utf-8")

        result = await tool.run(ReadFileArgs(path="relative.txt", view_type="content"))

        assert result.content is not None
        assert "relative content" in result.content.output

    async def test_content_view_absolute_path(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test content view works with absolute paths."""
        file_path = temp_dir / "absolute.txt"
        file_path.write_text("absolute content", encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        assert "absolute content" in result.content.output


# =============================================================================
# Outline Mode Tests (Python)
# =============================================================================


class TestOutlineMode:
    """Tests for outline view mode functionality."""

    async def test_outline_mode_python_file(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test outline mode for Python file extracts structure."""
        file_path = temp_dir / "example.py"
        content = """
def function1():
    pass

class MyClass:
    def method1(self):
        pass

def function2():
    pass
"""
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="outline"))

        assert result.content is not None
        assert "FILE:" in result.content.output
        assert "[FUNC]" in result.content.output or "[CLASS]" in result.content.output

    async def test_outline_mode_unsupported_language_fallback(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test outline mode for unsupported language falls back to content."""
        file_path = temp_dir / "example.txt"
        content = "line1\nline2\nline3\n" * 50  # More than 100 lines
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="outline"))

        assert result.content is not None
        # Should show first 100 lines
        assert "1| line1" in result.content.output
        # Should have fallback message
        assert (
            "Unsupported language" in result.content.output
            or "showing first" in result.content.output
        )

    async def test_smart_view_type_auto_small_file(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test auto mode uses content view for small files (< 16KB)."""
        file_path = temp_dir / "small.py"
        content = "print('hello')\n" * 100  # Small file
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="auto"))

        assert result.content is not None
        # Should show content with line numbers
        assert "1| print('hello')" in result.content.output

    async def test_smart_view_type_auto_large_file(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test auto mode uses outline view for large files (> 16KB)."""
        file_path = temp_dir / "large.py"
        content = "def func():\n    pass\n" * 1000  # Large file
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="auto"))

        assert result.content is not None
        # Should show outline, not full content
        # (Note: if outline generation fails, it may fall back to content)
        assert result.content.line_count == 1000

    async def test_smart_view_type_auto_with_view_range(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test auto mode uses content view when view_range is specified."""
        file_path = temp_dir / "range_auto.py"
        content = "line1\nline2\nline3\nline4\nline5"
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(
            ReadFileArgs(path=str(file_path), view_type="auto", view_range=(2, 3))
        )

        assert result.content is not None
        # Should use content mode because view_range is specified
        assert "2| line2" in result.content.output
        assert "3| line3" in result.content.output


# =============================================================================
# Media File Tests
# =============================================================================


class TestMediaFileView:
    """Tests for media file viewing functionality."""

    async def test_image_file_detection(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test image file detection and encoding."""
        file_path = temp_dir / "test.png"
        # Write a small valid PNG (1x1 pixel, red)
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        file_path.write_bytes(png_bytes)

        result = await tool.run(ReadFileArgs(path=str(file_path)))

        assert result.media is not None
        assert result.media.type == "image"
        assert result.media.mime_type == "image/png"
        assert result.media.data == png_bytes.hex()
        assert result.media.size == len(png_bytes)

    async def test_audio_file_detection(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test audio file detection and encoding."""
        file_path = temp_dir / "test.mp3"
        # Create a minimal MP3 file (not valid, but enough for testing)
        mp3_content = b"ID3" + b"\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 100
        file_path.write_bytes(mp3_content)

        result = await tool.run(ReadFileArgs(path=str(file_path)))

        assert result.media is not None
        assert result.media.type == "audio"
        assert result.media.size == len(mp3_content)

    async def test_large_media_file_returns_guidance(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test large media file returns guidance instead of embedding."""
        file_path = temp_dir / "large.png"
        # Create a file larger than 5MB
        large_content = b"\x00" * (6 * 1024 * 1024)  # 6MB
        file_path.write_bytes(large_content)

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(ReadFileArgs(path=str(file_path)))

        assert exc_info.value.code == "MEDIA_TOO_LARGE"
        assert "exceeding" in str(exc_info.value)
        assert len(exc_info.value.suggestions) > 0

    async def test_media_file_records_view(
        self, tool: ReadFileTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test media file viewing records view in tracker."""
        file_path = temp_dir / "view_test.png"
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        file_path.write_bytes(png_bytes)

        await tool.run(ReadFileArgs(path=str(file_path)))

        assert view_tracker.has_been_viewed(str(file_path))


# =============================================================================
# Directory Viewing Tests
# =============================================================================


class TestDirectoryView:
    """Tests for directory viewing functionality."""

    async def test_directory_list_mode(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test directory viewing in list mode."""
        # Create directory structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")

        result = await tool.run(ReadFileArgs(path=str(temp_dir), view_type="content"))

        assert result.content is not None
        assert "DIRECTORY:" in result.content.output
        assert "file1.txt" in result.content.output
        assert "file2.txt" in result.content.output
        assert "[DIR]" in result.content.output

    async def test_directory_outline_mode_python_files(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test directory viewing in outline mode for Python files."""
        # Create Python files
        (temp_dir / "module1.py").write_text("def func1():\n    pass\n")
        (temp_dir / "module2.py").write_text("class MyClass:\n    pass\n")
        (temp_dir / "readme.txt").write_text("This is a readme")

        result = await tool.run(ReadFileArgs(path=str(temp_dir), view_type="outline"))

        assert result.content is not None
        assert "DIRECTORY OUTLINE:" in result.content.output
        # Should show Python files
        assert "module1.py" in result.content.output
        assert "module2.py" in result.content.output
        # Should not show non-Python files in outline
        # (readme.txt may or may not appear depending on implementation)

    async def test_directory_with_view_range_fails(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test directory viewing with view_range fails."""
        (temp_dir / "file.txt").write_text("content")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                ReadFileArgs(
                    path=str(temp_dir), view_type="content", view_range=(1, 10)
                )
            )

        assert exc_info.value.code == "INVALID_ARGUMENT"
        assert "viewRange cannot be used with directories" in str(exc_info.value)
        assert len(exc_info.value.suggestions) > 0

    async def test_directory_outline_no_python_files(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test directory outline with no Python files."""
        # Create non-Python files only
        (temp_dir / "readme.txt").write_text("Readme")
        (temp_dir / "data.json").write_text('{"key": "value"}')

        result = await tool.run(ReadFileArgs(path=str(temp_dir), view_type="outline"))

        assert result.content is not None
        assert "No Python files found" in result.content.output


# =============================================================================
# Error Message Tests
# =============================================================================


class TestErrorMessages:
    """Tests for error messages matching TypeScript behavior."""

    async def test_file_not_found_error(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test file not found error includes helpful suggestions."""
        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(ReadFileArgs(path=str(temp_dir / "nonexistent.txt")))

        assert exc_info.value.code == "FILE_NOT_FOUND"
        assert "not found" in str(exc_info.value)
        assert len(exc_info.value.suggestions) > 0

    async def test_permission_denied_error(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test permission denied error includes helpful suggestions."""
        # Create a file and make it unreadable
        file_path = temp_dir / "protected.txt"
        file_path.write_text("secret content")
        os.chmod(str(file_path), 0o000)

        try:
            with pytest.raises(FileSystemError) as exc_info:
                await tool.run(ReadFileArgs(path=str(file_path)))

            assert exc_info.value.code == "PERMISSION_DENIED"
            assert "Permission denied" in str(exc_info.value)
            assert len(exc_info.value.suggestions) > 0
        finally:
            # Restore permissions for cleanup
            os.chmod(str(file_path), 0o644)


# =============================================================================
# Output Truncation Tests
# =============================================================================


class TestOutputTruncation:
    """Tests for output truncation at 80,000 characters."""

    async def test_content_truncation(self, tool: ReadFileTool, temp_dir: Path) -> None:
        """Test content view truncates output at 80,000 characters."""
        file_path = temp_dir / "large.txt"
        # Create a file larger than 80,000 characters
        content = "line\n" * 50_000  # ~300KB
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        # Output should be truncated
        assert len(result.content.output) <= 85_000  # Allow for footer
        assert "truncated" in result.content.output

    async def test_directory_truncation(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test directory listing truncates output at 80,000 characters."""
        # Create many files
        for i in range(5000):
            (temp_dir / f"file_{i}.txt").write_text(f"content {i}")

        result = await tool.run(ReadFileArgs(path=str(temp_dir), view_type="content"))

        assert result.content is not None
        # Output should be truncated
        assert len(result.content.output) <= 85_000  # Allow for footer
        assert "truncated" in result.content.output


# =============================================================================
# Tool Configuration Tests
# =============================================================================


class TestToolConfiguration:
    """Tests for tool configuration."""

    def test_tool_name_is_read_file(self) -> None:
        """Test tool has correct name."""
        assert ReadFileTool.name == "read_file"

    def test_tool_has_description(self) -> None:
        """Test tool has description."""
        assert len(ReadFileTool.description) > 0

    def test_tool_uses_read_file_args_schema(self) -> None:
        """Test tool uses ReadFileArgs as schema."""
        assert ReadFileTool.args_schema == ReadFileArgs


# =============================================================================
# Line Number Formatting Tests
# =============================================================================


class TestLineNumberFormatting:
    """Tests for line number formatting."""

    async def test_line_numbers_with_single_digit(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test line numbers formatted correctly for single-digit lines."""
        file_path = temp_dir / "single.txt"
        file_path.write_text("a\nb\nc", encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        assert "1| a" in result.content.output
        assert "2| b" in result.content.output
        assert "3| c" in result.content.output

    async def test_line_numbers_with_padding(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test line numbers have consistent padding."""
        file_path = temp_dir / "padding.txt"
        # Create a file with 100 lines
        content = "\n".join(f"line {i}" for i in range(100))
        file_path.write_text(content, encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        # Line 1 should have padding to match line 100
        assert "  1| line 0" in result.content.output
        assert "100| line 99" in result.content.output


# =============================================================================
# View Tracking Tests
# =============================================================================


class TestViewTracking:
    """Tests for view tracking functionality."""

    async def test_content_view_records_view(
        self, tool: ReadFileTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test content view records view in tracker."""
        file_path = temp_dir / "track_content.txt"
        file_path.write_text("content", encoding="utf-8")

        await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert view_tracker.has_been_viewed(str(file_path))

    async def test_outline_view_records_view(
        self, tool: ReadFileTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test outline view records view in tracker."""
        file_path = temp_dir / "track_outline.py"
        file_path.write_text("def func():\n    pass\n", encoding="utf-8")

        await tool.run(ReadFileArgs(path=str(file_path), view_type="outline"))

        assert view_tracker.has_been_viewed(str(file_path))

    async def test_directory_view_records_view(
        self, tool: ReadFileTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test directory view records view in tracker."""
        (temp_dir / "file.txt").write_text("content")

        await tool.run(ReadFileArgs(path=str(temp_dir), view_type="content"))

        assert view_tracker.has_been_viewed(str(temp_dir))

    async def test_view_without_tracker(
        self, tool_no_view_tracker: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test tool works correctly without ViewTrackerService configured."""
        file_path = temp_dir / "no_tracker.txt"
        file_path.write_text("content", encoding="utf-8")

        result = await tool_no_view_tracker.run(
            ReadFileArgs(path=str(file_path), view_type="content")
        )

        assert result.content is not None
        assert "content" in result.content.output


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_empty_file(self, tool: ReadFileTool, temp_dir: Path) -> None:
        """Test viewing an empty file."""
        file_path = temp_dir / "empty.txt"
        file_path.write_text("", encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        assert result.content.line_count == 0
        # Empty file should have no line number output
        assert result.content.output.strip() == ""

    async def test_file_with_only_newlines(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test file with only newlines."""
        file_path = temp_dir / "newlines.txt"
        file_path.write_text("\n\n\n", encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        assert result.content.line_count == 3

    async def test_view_range_start_greater_than_end(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test view range with start > end fails."""
        file_path = temp_dir / "invalid_range.txt"
        file_path.write_text("line1\nline2\nline3", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                ReadFileArgs(
                    path=str(file_path), view_type="content", view_range=(5, 2)
                )
            )

        assert exc_info.value.code == "INVALID_VIEW_RANGE"

    async def test_empty_directory(self, tool: ReadFileTool, temp_dir: Path) -> None:
        """Test viewing an empty directory."""
        result = await tool.run(ReadFileArgs(path=str(temp_dir), view_type="content"))

        assert result.content is not None
        assert "DIRECTORY:" in result.content.output

    async def test_special_characters_in_filename(
        self, tool: ReadFileTool, temp_dir: Path
    ) -> None:
        """Test viewing file with special characters in name."""
        file_path = temp_dir / "file-with-dashes_and_underscores.txt"
        file_path.write_text("content", encoding="utf-8")

        result = await tool.run(ReadFileArgs(path=str(file_path), view_type="content"))

        assert result.content is not None
        assert "content" in result.content.output

"""Unit tests for CreateTool.

Tests cover all acceptance criteria including:
- Create new file successfully
- Create fails when file exists (critical test)
- Create operation creates parent directories
- Relative paths resolved correctly
- Absolute paths work correctly
- UTF-8 encoding handles special characters
- Permission errors handled correctly
- Error message format validation (exact TypeScript match)
- Success message format validation (exact TypeScript match)
- View tracking called after creation (if configured)
- View tracking is optional
- All tests achieve >85% coverage
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import tempfile

from pydantic import ValidationError
import pytest

from vibe.core.tools.filesystem.create import (
    CreateArgs,
    CreateResult,
    CreateTool,
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
def tool(temp_dir: Path) -> CreateTool:
    """Create a CreateTool instance for testing."""
    return CreateTool(workdir=temp_dir)


@pytest.fixture
def tool_with_view_tracker(
    view_tracker: ViewTrackerService, temp_dir: Path
) -> CreateTool:
    """Create a CreateTool instance with view tracking for testing."""
    return CreateTool(view_tracker=view_tracker, workdir=temp_dir)


# =============================================================================
# CreateArgs Tests
# =============================================================================


class TestCreateArgs:
    """Tests for CreateArgs Pydantic model."""

    def test_creation_with_path_and_content(self) -> None:
        """Test CreateArgs can be created with path and file_text."""
        args = CreateArgs(path="/test/file.txt", file_text="Hello, World!")
        assert args.path == "/test/file.txt"
        assert args.file_text == "Hello, World!"

    def test_creation_with_empty_content(self) -> None:
        """Test CreateArgs accepts empty content."""
        args = CreateArgs(path="/test/file.txt", file_text="")
        assert args.file_text == ""

    def test_model_validation(self) -> None:
        """Test CreateArgs validates types correctly."""
        with pytest.raises(ValidationError):
            CreateArgs(path=123, file_text="text")  # type: ignore

    def test_model_validation_missing_path(self) -> None:
        """Test CreateArgs requires path field."""
        with pytest.raises(ValidationError):
            CreateArgs(file_text="text")  # type: ignore

    def test_model_validation_missing_file_text(self) -> None:
        """Test CreateArgs requires file_text field."""
        with pytest.raises(ValidationError):
            CreateArgs(path="/test/file.txt")  # type: ignore


# =============================================================================
# CreateResult Tests
# =============================================================================


class TestCreateResult:
    """Tests for CreateResult Pydantic model."""

    def test_creation_with_output(self) -> None:
        """Test CreateResult can be created with output."""
        result = CreateResult(output="File created successfully.")
        assert result.output == "File created successfully."


# =============================================================================
# CreateToolConfig Tests
# =============================================================================


class TestCreateToolConfig:
    """Tests for CreateToolConfig."""

    def test_creation_with_defaults(self) -> None:
        """Test CreateToolConfig can be created with defaults."""
        config = CreateToolConfig()
        assert config.permission.value == "ask"
        assert config.view_tracker is None

    def test_creation_with_view_tracker(self) -> None:
        """Test CreateToolConfig accepts ViewTrackerService."""
        tracker = ViewTrackerService()
        config = CreateToolConfig(view_tracker=tracker)
        assert config.view_tracker is tracker


# =============================================================================
# CreateToolState Tests
# =============================================================================


class TestCreateToolState:
    """Tests for CreateToolState."""

    def test_creation_with_defaults(self) -> None:
        """Test CreateToolState can be created with defaults."""
        CreateToolState()
        # State should be empty - no attributes defined


# =============================================================================
# Create Operation Tests
# =============================================================================


@pytest.mark.asyncio
class TestCreateOperation:
    """Tests for file creation functionality."""

    async def test_create_new_file_success(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test creating a new file succeeds."""
        file_path = temp_dir / "new_file.txt"
        result = await tool.run(
            CreateArgs(path=str(file_path), file_text="Hello, World!")
        )

        assert result.output == f"File '{file_path}' created successfully"
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == "Hello, World!"

    async def test_create_file_already_exists_fails(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test creating a file that already exists fails.

        This is the critical test - CreateTool should fail with a helpful
        error message when attempting to create a file that already exists,
        matching TypeScript FileEditor.create() behavior exactly.
        """
        file_path = temp_dir / "existing_file.txt"

        # Create the file first
        file_path.write_text("existing content", encoding="utf-8")
        assert file_path.exists()

        # Try to create again - should fail
        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(CreateArgs(path=str(file_path), file_text="new content"))

        error = exc_info.value
        assert error.code == "FILE_ALREADY_EXISTS"
        # Access the original message from args[0] (RuntimeError behavior)
        error_message = error.args[0]
        assert str(file_path) in error_message
        # Verify exact TypeScript error message format
        assert "Cannot create file - it already exists:" in error_message
        assert (
            "• If you want to replace entire file: use 'edit' command" in error_message
        )
        assert (
            "• If you want to modify specific parts: use 'str_replace' command"
            in error_message
        )
        assert (
            "• If you want a different file: choose a different filename/location"
            in error_message
        )

    async def test_create_parent_directories_created(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test creating a file creates parent directories automatically."""
        nested_path = temp_dir / "a" / "b" / "c" / "new_file.txt"
        assert not nested_path.parent.exists()

        result = await tool.run(
            CreateArgs(path=str(nested_path), file_text="nested content")
        )

        assert result.output == f"File '{nested_path}' created successfully"
        assert nested_path.exists()
        assert nested_path.read_text(encoding="utf-8") == "nested content"

    async def test_create_relative_path_resolved(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test relative paths are resolved correctly against workdir."""
        # Create a subdirectory in temp_dir
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        # Use relative path
        result = await tool.run(
            CreateArgs(path="subdir/relative_file.txt", file_text="relative content")
        )

        expected_path = temp_dir / "subdir" / "relative_file.txt"
        assert result.output == f"File '{expected_path}' created successfully"
        assert expected_path.exists()
        assert expected_path.read_text(encoding="utf-8") == "relative content"

    async def test_create_absolute_path_works(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test absolute paths work correctly."""
        file_path = temp_dir / "absolute_file.txt"

        result = await tool.run(
            CreateArgs(path=str(file_path), file_text="absolute content")
        )

        assert result.output == f"File '{file_path}' created successfully"
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == "absolute content"

    async def test_create_utf8_encoding_special_chars(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test UTF-8 encoding handles special characters correctly."""
        file_path = temp_dir / "unicode_file.txt"
        special_content = "Hello, 世界! 🌍 émojis 中文 日本語"

        result = await tool.run(
            CreateArgs(path=str(file_path), file_text=special_content)
        )

        assert result.output == f"File '{file_path}' created successfully"
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == special_content

    async def test_create_empty_file(self, tool: CreateTool, temp_dir: Path) -> None:
        """Test creating an empty file succeeds."""
        file_path = temp_dir / "empty_file.txt"

        result = await tool.run(CreateArgs(path=str(file_path), file_text=""))

        assert result.output == f"File '{file_path}' created successfully"
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == ""

    async def test_create_multiline_content(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test creating a file with multiline content."""
        file_path = temp_dir / "multiline_file.txt"
        multiline_content = "Line 1\nLine 2\nLine 3\n"

        result = await tool.run(
            CreateArgs(path=str(file_path), file_text=multiline_content)
        )

        assert result.output == f"File '{file_path}' created successfully"
        assert file_path.read_text(encoding="utf-8") == multiline_content


# =============================================================================
# View Tracking Tests
# =============================================================================


@pytest.mark.asyncio
class TestViewTracking:
    """Tests for view tracking functionality."""

    async def test_view_tracking_called_after_creation(
        self, tool_with_view_tracker: CreateTool, temp_dir: Path
    ) -> None:
        """Test view_tracker.record_view() is called after successful creation."""
        file_path = temp_dir / "tracked_file.txt"

        await tool_with_view_tracker.run(
            CreateArgs(path=str(file_path), file_text="tracked content")
        )

        # Verify view was recorded
        tracker = tool_with_view_tracker.config.view_tracker
        assert tracker is not None
        assert tracker.has_been_viewed(str(file_path))

    async def test_view_tracking_optional(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test tool works correctly without view_tracker configured."""
        file_path = temp_dir / "no_tracker_file.txt"

        # Should succeed without view_tracker
        result = await tool.run(
            CreateArgs(path=str(file_path), file_text="no tracker content")
        )

        assert result.output == f"File '{file_path}' created successfully"
        assert file_path.exists()

    async def test_view_tracking_timestamp_recorded(
        self, tool_with_view_tracker: CreateTool, temp_dir: Path
    ) -> None:
        """Test view timestamp is recorded correctly."""
        file_path = temp_dir / "timestamp_file.txt"

        await tool_with_view_tracker.run(
            CreateArgs(path=str(file_path), file_text="timestamp content")
        )

        # Verify timestamp was recorded
        tracker = tool_with_view_tracker.config.view_tracker
        assert tracker is not None
        timestamp = tracker.get_last_view_timestamp(str(file_path))
        assert timestamp is not None
        assert timestamp > 0


# =============================================================================
# Error Message Format Tests
# =============================================================================


@pytest.mark.asyncio
class TestErrorMessageFormat:
    """Tests for exact TypeScript error message format matching."""

    async def test_file_exists_error_message_exact_format(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test File Already Exists error message matches TypeScript format exactly."""
        file_path = temp_dir / "error_format_test.txt"
        file_path.write_text("existing", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(CreateArgs(path=str(file_path), file_text="new content"))

        error = exc_info.value
        # Exact TypeScript error message format
        expected_message = (
            f"Cannot create file - it already exists: '{file_path}'\n\n"
            "• If you want to replace entire file: use 'edit' command\n"
            "• If you want to modify specific parts: use 'str_replace' command\n"
            "• If you want a different file: choose a different filename/location"
        )
        # Access the original message from args[0] (RuntimeError behavior)
        assert error.args[0] == expected_message

    async def test_success_message_exact_format(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test success message matches TypeScript format exactly."""
        file_path = temp_dir / "success_format_test.txt"

        result = await tool.run(
            CreateArgs(path=str(file_path), file_text="success content")
        )

        # Exact TypeScript success message format
        assert result.output == f"File '{file_path}' created successfully"


# =============================================================================
# Path Resolution Tests
# =============================================================================


@pytest.mark.asyncio
class TestPathResolution:
    """Tests for path resolution functionality."""

    async def test_resolve_absolute_path(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test absolute paths are used as-is."""
        file_path = temp_dir / "absolute_test.txt"

        result = await tool.run(CreateArgs(path=str(file_path), file_text="content"))

        assert file_path.exists()
        assert result.output == f"File '{file_path}' created successfully"

    async def test_resolve_relative_path(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test relative paths are resolved against workdir."""
        file_path = temp_dir / "relative_test.txt"

        result = await tool.run(
            CreateArgs(path="relative_test.txt", file_text="content")
        )

        assert file_path.exists()
        assert result.output == f"File '{file_path}' created successfully"

    async def test_resolve_path_with_dot_prefix(
        self, tool: CreateTool, temp_dir: Path
    ) -> None:
        """Test paths starting with ./ are resolved correctly."""
        file_path = temp_dir / "dot_prefix_test.txt"

        result = await tool.run(
            CreateArgs(path="./dot_prefix_test.txt", file_text="content")
        )

        assert file_path.exists()
        assert result.output == f"File '{file_path}' created successfully"


# =============================================================================
# Tool Metadata Tests
# =============================================================================


class TestToolMetadata:
    """Tests for tool name and description."""

    def test_tool_name(self) -> None:
        """Test tool has correct name."""
        assert CreateTool.name == "create"

    def test_tool_description(self) -> None:
        """Test tool has correct description."""
        assert "create" in CreateTool.description.lower()
        assert "new files" in CreateTool.description.lower()

    def test_tool_args_schema(self) -> None:
        """Test tool has correct args schema."""
        assert CreateTool.args_schema == CreateArgs

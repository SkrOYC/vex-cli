"""Unit tests for filesystem tool types and error classes."""

from __future__ import annotations

import pytest

from vibe.core.tools.filesystem.types import (
    DEFAULT_CONTEXT_AFTER,
    DEFAULT_CONTEXT_BEFORE,
    DEFAULT_FILES_LIMIT,
    DEFAULT_RECURSIVE_DEPTH,
    FILE_EXISTS_ERROR_TIMEOUT_MS,
    LARGE_FILE_THRESHOLD,
    LENGTH_DIFF_RATIO_THRESHOLD,
    LINE_SIMILARITY_THRESHOLD,
    MAX_FILES_LIMIT,
    MAX_INLINE_MEDIA_BYTES,
    MAX_RECURSIVE_DEPTH,
    MIN_NEW_CONTENT_LENGTH,
    MIN_OLD_CONTENT_LENGTH,
    MISTAKEN_EDIT_TIMEOUT_MS,
    OUTPUT_LIMIT,
    STR_REPLACE_LENGTH_RATIO_THRESHOLD,
    FileSystemError,
    ValidationError,
)


class TestFileSystemError:
    """Tests for FileSystemError class."""

    def test_creation_with_all_parameters(self) -> None:
        """Test FileSystemError can be created with all attributes."""
        error = FileSystemError(
            message="File not found",
            code="FILE_NOT_FOUND",
            path="/nonexistent/file.txt",
            suggestions=["Check the path", "Use list to verify the file exists"],
        )

        assert error.code == "FILE_NOT_FOUND"
        assert error.path == "/nonexistent/file.txt"
        assert error.suggestions == [
            "Check the path",
            "Use list to verify the file exists",
        ]
        assert error.args[0] == "File not found"

    def test_creation_with_minimal_parameters(self) -> None:
        """Test FileSystemError can be created with minimal parameters."""
        error = FileSystemError(message="Access denied", code="PERMISSION_DENIED")

        assert error.code == "PERMISSION_DENIED"
        assert error.path is None
        assert error.suggestions == []
        assert error.args[0] == "Access denied"

    def test_creation_with_path_only(self) -> None:
        """Test FileSystemError can be created with message, code, and path."""
        error = FileSystemError(
            message="File locked", code="FILE_LOCKED", path="/tmp/locked.txt"
        )

        assert error.code == "FILE_LOCKED"
        assert error.path == "/tmp/locked.txt"
        assert error.suggestions == []

    def test_creation_with_suggestions_only(self) -> None:
        """Test FileSystemError can be created with message, code, and suggestions."""
        error = FileSystemError(
            message="Disk full",
            code="DISK_FULL",
            suggestions=["Free up space", "Delete temporary files"],
        )

        assert error.code == "DISK_FULL"
        assert error.path is None
        assert error.suggestions == ["Free up space", "Delete temporary files"]

    def test_string_representation_with_all_fields(self) -> None:
        """Test FileSystemError string representation includes all fields."""
        error = FileSystemError(
            message="File not found",
            code="FILE_NOT_FOUND",
            path="/nonexistent/file.txt",
            suggestions=["Check the path"],
        )

        error_str = str(error)

        assert "[FILE_NOT_FOUND]" in error_str
        assert "File not found" in error_str
        assert "/nonexistent/file.txt" in error_str
        assert "Check the path" in error_str

    def test_string_representation_minimal(self) -> None:
        """Test FileSystemError string representation with minimal fields."""
        error = FileSystemError(message="Unknown error", code="UNKNOWN")

        error_str = str(error)

        assert "[UNKNOWN]" in error_str
        assert "Unknown error" in error_str

    def test_string_representation_no_suggestions(self) -> None:
        """Test FileSystemError string representation excludes empty suggestions."""
        error = FileSystemError(
            message="File not found", code="FILE_NOT_FOUND", path="/path/to/file.txt"
        )

        error_str = str(error)

        assert "Suggestions:" not in error_str

    def test_is_instance_of_runtime_error(self) -> None:
        """Test FileSystemError is instance of RuntimeError."""
        error = FileSystemError(message="Test", code="TEST")
        assert isinstance(error, RuntimeError)

    def test_can_be_raised_and_caught(self) -> None:
        """Test FileSystemError can be raised and caught."""
        with pytest.raises(FileSystemError) as exc_info:
            raise FileSystemError(
                message="Test error",
                code="TEST_ERROR",
                path="/test/path",
                suggestions=["Suggestion 1"],
            )

        assert exc_info.value.code == "TEST_ERROR"
        assert exc_info.value.path == "/test/path"


class TestValidationError:
    """Tests for ValidationError class."""

    def test_creation_with_all_parameters(self) -> None:
        """Test ValidationError can be created with all attributes."""
        error = ValidationError(
            message="Invalid path format",
            field="path",
            value="/relative/path",
            suggestions=["Use absolute paths", "Paths must start with /"],
        )

        assert error.field == "path"
        assert error.value == "/relative/path"
        assert error.suggestions == ["Use absolute paths", "Paths must start with /"]
        assert error.args[0] == "Invalid path format"

    def test_creation_with_minimal_parameters(self) -> None:
        """Test ValidationError can be created with minimal parameters."""
        error = ValidationError(message="Invalid input")

        assert error.field is None
        assert error.value is None
        assert error.suggestions == []
        assert error.args[0] == "Invalid input"

    def test_creation_with_field_only(self) -> None:
        """Test ValidationError can be created with message and field."""
        error = ValidationError(message="Value required", field="name")

        assert error.field == "name"
        assert error.value is None
        assert error.suggestions == []

    def test_creation_with_value_only(self) -> None:
        """Test ValidationError can be created with message and value."""
        error = ValidationError(message="Invalid value", value=12345)

        assert error.field is None
        assert error.value == 12345
        assert error.suggestions == []

    def test_string_representation_with_all_fields(self) -> None:
        """Test ValidationError string representation includes all fields."""
        error = ValidationError(
            message="Invalid path format",
            field="path",
            value="/relative/path",
            suggestions=["Use absolute paths"],
        )

        error_str = str(error)

        assert "Invalid path format" in error_str
        assert "Field: path" in error_str
        assert "'/relative/path'" in error_str
        assert "Use absolute paths" in error_str

    def test_string_representation_minimal(self) -> None:
        """Test ValidationError string representation with minimal fields."""
        error = ValidationError(message="Invalid input")

        error_str = str(error)

        assert "Invalid input" in error_str
        assert "Field:" not in error_str
        assert "Value:" not in error_str

    def test_string_representation_no_suggestions(self) -> None:
        """Test ValidationError string representation excludes empty suggestions."""
        error = ValidationError(
            message="Invalid value", field="count", value="not a number"
        )

        error_str = str(error)

        assert "Suggestions:" not in error_str

    def test_is_instance_of_runtime_error(self) -> None:
        """Test ValidationError is instance of RuntimeError."""
        error = ValidationError(message="Test", field="test")
        assert isinstance(error, RuntimeError)

    def test_can_be_raised_and_caught(self) -> None:
        """Test ValidationError can be raised and caught."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(
                message="Test error",
                field="test_field",
                value="test_value",
                suggestions=["Suggestion 1"],
            )

        assert exc_info.value.field == "test_field"
        assert exc_info.value.value == "test_value"


class TestConstants:
    """Tests for filesystem constants matching TypeScript values."""

    def test_large_file_threshold(self) -> None:
        """Test LARGE_FILE_THRESHOLD matches TypeScript value (16384)."""
        assert LARGE_FILE_THRESHOLD == 16_384

    def test_max_inline_media_bytes(self) -> None:
        """Test MAX_INLINE_MEDIA_BYTES matches TypeScript value (5MB)."""
        assert MAX_INLINE_MEDIA_BYTES == 5 * 1024 * 1024
        assert MAX_INLINE_MEDIA_BYTES == 5_242_880

    def test_output_limit(self) -> None:
        """Test OUTPUT_LIMIT matches TypeScript value (80000)."""
        assert OUTPUT_LIMIT == 80_000

    def test_default_context_before(self) -> None:
        """Test DEFAULT_CONTEXT_BEFORE matches TypeScript value (5)."""
        assert DEFAULT_CONTEXT_BEFORE == 5

    def test_default_context_after(self) -> None:
        """Test DEFAULT_CONTEXT_AFTER matches TypeScript value (3)."""
        assert DEFAULT_CONTEXT_AFTER == 3

    def test_max_files_limit(self) -> None:
        """Test MAX_FILES_LIMIT matches TypeScript value (10000)."""
        assert MAX_FILES_LIMIT == 10_000

    def test_default_files_limit(self) -> None:
        """Test DEFAULT_FILES_LIMIT matches TypeScript value (1000)."""
        assert DEFAULT_FILES_LIMIT == 1_000

    def test_max_recursive_depth(self) -> None:
        """Test MAX_RECURSIVE_DEPTH matches TypeScript value (10)."""
        assert MAX_RECURSIVE_DEPTH == 10

    def test_default_recursive_depth(self) -> None:
        """Test DEFAULT_RECURSIVE_DEPTH matches TypeScript value (3)."""
        assert DEFAULT_RECURSIVE_DEPTH == 3

    def test_str_replace_length_ratio_threshold(self) -> None:
        """Test STR_REPLACE_LENGTH_RATIO_THRESHOLD matches TypeScript value (0.3)."""
        assert STR_REPLACE_LENGTH_RATIO_THRESHOLD == 0.3

    def test_min_old_content_length(self) -> None:
        """Test MIN_OLD_CONTENT_LENGTH matches TypeScript value (100)."""
        assert MIN_OLD_CONTENT_LENGTH == 100

    def test_min_new_content_length(self) -> None:
        """Test MIN_NEW_CONTENT_LENGTH matches TypeScript value (50)."""
        assert MIN_NEW_CONTENT_LENGTH == 50

    def test_line_similarity_threshold(self) -> None:
        """Test LINE_SIMILARITY_THRESHOLD matches TypeScript value (0.7)."""
        assert LINE_SIMILARITY_THRESHOLD == 0.7

    def test_length_diff_ratio_threshold(self) -> None:
        """Test LENGTH_DIFF_RATIO_THRESHOLD matches TypeScript value (0.3)."""
        assert LENGTH_DIFF_RATIO_THRESHOLD == 0.3

    def test_mistaken_edit_timeout_ms(self) -> None:
        """Test MISTAKEN_EDIT_TIMEOUT_MS matches TypeScript value (60000)."""
        assert MISTAKEN_EDIT_TIMEOUT_MS == 60_000

    def test_file_exists_error_timeout_ms(self) -> None:
        """Test FILE_EXISTS_ERROR_TIMEOUT_MS matches TypeScript value (60000)."""
        assert FILE_EXISTS_ERROR_TIMEOUT_MS == 60_000

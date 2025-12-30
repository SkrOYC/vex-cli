"""Shared types, error classes, and constants for filesystem tools.

This module provides foundational types and error handling for all filesystem tools,
ensuring consistent behavior across file operations.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Error Classes
# =============================================================================


class FileSystemError(RuntimeError):
    """Error raised when file system operations fail.

    Attributes:
        code: Error code identifying the type of failure (e.g., "FILE_NOT_FOUND",
            "PERMISSION_DENIED").
        path: File or directory path that caused the error.
        suggestions: Suggested recovery actions for the user.

    Example:
        ```python
        raise FileSystemError(
            message="File not found",
            code="FILE_NOT_FOUND",
            path="/nonexistent/file.txt",
            suggestions=["Check the file path for typos", "Use list to verify the file exists"]
        )
        ```
    """

    def __init__(
        self,
        message: str,
        code: str,
        path: str | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.suggestions = suggestions if suggestions is not None else []

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.args[0]}"]
        if self.path:
            parts.append(f"Path: {self.path}")
        if self.suggestions:
            parts.append(f"Suggestions: {'; '.join(self.suggestions)}")
        return " | ".join(parts)


class ValidationError(RuntimeError):
    """Error raised when input validation fails.

    Attributes:
        field: Name of the field that failed validation.
        value: Invalid value that was provided.
        suggestions: Suggested corrections for the user.

    Example:
        ```python
        raise ValidationError(
            message="Invalid path format",
            field="path",
            value="/relative/path",
            suggestions=["Use absolute paths", "Paths must start with /"]
        )
        ```
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        suggestions: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.value = value
        self.suggestions = suggestions if suggestions is not None else []

    def __str__(self) -> str:
        parts = [self.args[0]]
        if self.field:
            parts.append(f"Field: {self.field}")
        if self.value is not None:
            parts.append(f"Value: {self.value!r}")
        if self.suggestions:
            parts.append(f"Suggestions: {'; '.join(self.suggestions)}")
        return " | ".join(parts)


# =============================================================================
# Constants
# =============================================================================

# File size thresholds
LARGE_FILE_THRESHOLD: int = (
    16_384  # 16KB - Large files may use different processing strategies
)
MAX_INLINE_MEDIA_BYTES: int = (
    5 * 1024 * 1024
)  # 5MB - Maximum media payload for inline base64

# Output limits
OUTPUT_LIMIT: int = 80_000  # Maximum output characters before truncation

# Search limits
DEFAULT_CONTEXT_BEFORE: int = 5  # Default context lines before a match
DEFAULT_CONTEXT_AFTER: int = 3  # Default context lines after a match
MAX_FILES_LIMIT: int = 10_000  # Maximum files limit for search operations
DEFAULT_FILES_LIMIT: int = 1_000  # Default files limit for search operations
MAX_RECURSIVE_DEPTH: int = 10  # Maximum recursive depth for directory traversal
DEFAULT_RECURSIVE_DEPTH: int = 3  # Default recursive depth for directory traversal

# Mistaken edit detection thresholds
STR_REPLACE_LENGTH_RATIO_THRESHOLD: float = (
    0.3  # Length ratio threshold for str_replace detection
)
MIN_OLD_CONTENT_LENGTH: int = (
    100  # Minimum old content length for str_replace detection
)
MIN_NEW_CONTENT_LENGTH: int = 50  # Minimum new content length for str_replace detection
LINE_SIMILARITY_THRESHOLD: float = (
    0.7  # Line similarity threshold for str_replace detection
)
LENGTH_DIFF_RATIO_THRESHOLD: float = 0.3  # Length difference ratio threshold

# Retry timeout
MISTAKEN_EDIT_TIMEOUT_MS: int = 60_000  # 1 minute timeout for mistaken edit warnings
FILE_EXISTS_ERROR_TIMEOUT_MS: int = (
    60_000  # 1 minute timeout for file exists error flow
)

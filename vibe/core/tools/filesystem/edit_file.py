"""EditFileTool implementation for precise string replacement in files.

This module provides the EditFileTool class that enables safe, targeted string
replacement in files with exact matching, validation, and helpful error messages.

Features:
- Exact string matching (must appear exactly once)
- Multiple match detection with helpful error messages
- File not found detection with helpful error messages
- UTF-8 encoding support
- Edit history tracking for undo functionality
- View tracking integration (optional)

Example:
    ```python
    from vibe.core.tools.filesystem.edit_file import EditFileTool

    tool = EditFileTool(workdir=Path("/project"))
    result = await tool.arun(path="test.py", old_str="hello", new_str="hi")
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from vibe.core.tools.base import BaseTool
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError

# Constant for exact match validation
_EXACT_MATCH_PARTS_COUNT: int = 2  # parts length when old_str appears exactly once

# =============================================================================
# Argument and Result Models
# =============================================================================


class EditFileArgs(BaseModel):
    """Arguments for the edit_file tool.

    Attributes:
        path: File path (absolute or relative to working directory).
        old_str: Exact string to replace in the file.
        new_str: Replacement string to insert.
    """

    path: str
    old_str: str
    new_str: str


class EditFileResult(BaseModel):
    """Result of the edit_file tool operation.

    Attributes:
        output: Success or error message describing the operation result.
    """

    output: str


# =============================================================================
# EditFileTool Implementation
# =============================================================================


class EditFileTool(BaseTool):
    """Tool for performing precise string replacements in files.

    This tool provides safe string replacement operations with multiple safety features:
    - Enforces exact string matching (must appear exactly once)
    - Detects and reports multiple occurrences with helpful guidance
    - Provides helpful error messages when text is not found
    - Tracks edit history for potential undo functionality
    - Supports UTF-8 encoding for file operations
    - Integrates with ViewTrackerService for view tracking (optional)
    """

    def __init__(
        self,
        view_tracker: ViewTrackerService | None = None,
        workdir: Path | None = None,
    ) -> None:
        """Initialize EditFileTool.

        Args:
            view_tracker: Optional service for tracking file views.
            workdir: Working directory for path resolution. Defaults to cwd if None.
        """
        super().__init__(
            name="edit_file",
            description="Replace a specific string in a file with exact matching",
            args_schema=EditFileArgs,
        )
        self._view_tracker = view_tracker
        self._workdir = workdir or Path.cwd()
        self._edit_history: dict[str, list[str]] = {}

    async def _arun(
        self,
        path: str,
        old_str: str,
        new_str: str,
    ) -> EditFileResult:
        """Execute the edit_file tool operation.

        Performs precise string replacement in the specified file. The old string
        must appear exactly once in the file for the replacement to succeed.

        Args:
            path: File path (absolute or relative to working directory).
            old_str: Exact string to replace in the file.
            new_str: Replacement string to insert.

        Returns:
            EditFileResult with success or error message.

        Raises:
            FileSystemError: If path resolution, file operations, or validation fails.
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(path)

        # Validate that old_str is not empty (split with empty separator is invalid)
        if not old_str:
            raise FileSystemError(
                message="The 'old_str' argument cannot be an empty string.",
                code="INVALID_ARGUMENT",
                path=str(resolved_path),
                suggestions=["Provide a non-empty string to search for."],
            )

        # Check if file exists
        if not resolved_path.exists():
            raise FileSystemError(
                message=f"""File not found: '{resolved_path}'

The specified file doesn't exist. Use 'create' command to create a new file, or check if the path is correct.""",
                code="FILE_NOT_FOUND",
                path=str(resolved_path),
                suggestions=[
                    "Use 'create' command to create a new file",
                    "Check the file path for typos or errors",
                    "Use 'list' command to verify the file exists",
                ],
            )

        # Read current content
        old_content = resolved_path.read_text(encoding="utf-8")

        # Split content by old_str to check for exact match
        parts = old_content.split(old_str)

        # Validate that old_str appears exactly once
        if len(parts) == 1:
            raise FileSystemError(
                message=f"""Text not found in '{resolved_path}'

The specified text '{old_str}' was not found in the file. Check for:
• Typos or whitespace differences
• Extra character escaping (content must exactly match the file)
• Use 'read_file' command to copy the precise text including all whitespace and formatting""",
                code="TEXT_NOT_FOUND",
                path=str(resolved_path),
                suggestions=[
                    "Check for typos or whitespace differences",
                    "Check for extra character escaping (content must exactly match the file)",
                    "Use 'read_file' command to copy the precise text including all whitespace and formatting",
                ],
            )

        if len(parts) > _EXACT_MATCH_PARTS_COUNT:
            raise FileSystemError(
                message=f"""Multiple matches found: '{old_str}' appears {len(parts) - 1} times in '{resolved_path}'

str_replace requires exactly one occurrence. To fix:
• Add more surrounding context to old_str to make it unique
• Include 3+ lines before and after target text
• Use 'create' command to create a new file or 'edit' to replace entire content""",
                code="MULTIPLE_MATCHES",
                path=str(resolved_path),
                suggestions=[
                    "Add more surrounding context to old_str to make it unique",
                    "Include 3+ lines before and after target text",
                    "Use 'create' command to create a new file or 'edit' to replace entire content",
                ],
            )

        # Save current content to edit history before modification
        self._push_history(str(resolved_path), old_content)

        # Perform the replacement using join
        new_content = new_str.join(parts)

        # Write the new content back to the file
        resolved_path.write_text(new_content, encoding="utf-8")

        # Record view after successful edit if view_tracker is configured
        if self._view_tracker is not None:
            self._view_tracker.record_view(str(resolved_path))

        return EditFileResult(output=f"File '{resolved_path}' modified successfully.")

    # =============================================================================
    # Path Resolution
    # =============================================================================

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative paths against working directory.

        Args:
            path: File path (absolute or relative).

        Returns:
            Absolute Path resolved against working directory.
        """
        if Path(path).is_absolute():
            return Path(path).resolve()
        else:
            return (self._workdir / path).resolve()

    # =============================================================================
    # Edit History Management
    # =============================================================================

    def _push_history(self, file_path: str, content: str) -> None:
        """Save the current file content to the edit history.

        Stores the current content of a file in the in-memory history stack.
        This enables undo functionality by maintaining a stack of previous
        versions for each file.

        Args:
            file_path: Absolute path to the file being modified.
            content: Current content of the file to save.
        """
        history = self._edit_history.setdefault(file_path, [])
        history.append(content)

    def _pop_history(self, file_path: str) -> str | None:
        """Pop the most recent content from the edit history.

        Retrieves and removes the last saved version of a file from the
        edit history stack.

        Args:
            file_path: Absolute path to the file.

        Returns:
            The previous content of the file, or None if no history exists.
        """
        history = self._edit_history.get(file_path)
        if not history:
            return None
        return history.pop()

    def _run(self, **kwargs: Any) -> str:
        """Synchronous execution not supported."""
        raise NotImplementedError("EditFileTool only supports async execution")

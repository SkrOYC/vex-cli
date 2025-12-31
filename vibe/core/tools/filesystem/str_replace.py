"""StrReplaceTool implementation for LangChain 1.2.0 integration.

This module provides StrReplaceTool class for precise string replacement
in files with exact matching and validation.

Features:
- Exact string matching (must appear exactly once)
- Multiple match detection with helpful error messages
- UTF-8 encoding support
- Edit history tracking for undo functionality
- View tracking integration
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.filesystem.langchain_base import VibeLangChainTool
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError


# =============================================================================
# Argument and Result Models
# =============================================================================


class StrReplaceArgs(BaseModel):
    """Arguments for str_replace tool.

    Attributes:
        path: File path (absolute or relative to working directory).
        old_str: Exact string to replace in file.
        new_str: Replacement string to insert.
    """

    path: str
    old_str: str
    new_str: str


class StrReplaceResult(BaseModel):
    """Result of str_replace tool operation.

    Attributes:
        output: Success or error message describing operation result.
    """

    output: str


# =============================================================================
# StrReplaceTool Implementation
# =============================================================================


class StrReplaceTool(VibeLangChainTool):
    """Tool for performing precise string replacements in files.

    This tool provides safe string replacement operations with multiple safety features:
    - Enforces exact string matching (must appear exactly once)
    - Detects and reports multiple occurrences with helpful guidance
    - Provides helpful error messages when text is not found
    - Tracks edit history for potential undo functionality
    - Supports UTF-8 encoding for file operations
    - Integrates with ViewTrackerService for view tracking (optional)

    Attributes:
        name: Tool name identifier.
        description: Description of tool functionality.
        args_schema: Pydantic model for input validation.
    """

    name = "str_replace"
    description = "Replace a specific string in a file with exact matching"

    def __init__(
        self,
        permission=Field(default=ToolPermission.ASK, exclude=True),
        view_tracker=None,
        workdir=None,
    ) -> None:
        """Initialize StrReplaceTool with VibeLangChainTool base."""
        super().__init__(
            permission=permission, view_tracker=view_tracker, workdir=workdir
        )

    def _run(self, input: dict[str, Any]) -> str:
        """Execute str_replace tool operation.

        Performs precise string replacement in specified file. The old string
        must appear exactly once in file for replacement to succeed.

        Args:
            input: Dictionary containing 'path', 'old_str', and 'new_str' keys.

        Returns:
            Success message or error message describing the result.

        Raises:
            FileSystemError: If validation fails or file operations encounter errors.
        """
        # Validate that old_str is not empty (split with empty separator is invalid)
        old_str = input.get("old_str", "")
        if not old_str:
            raise FileSystemError(
                message="The 'old_str' argument cannot be an empty string.",
                code="INVALID_ARGUMENT",
                path=input.get("path"),
                suggestions=["Provide a non-empty string to search for."],
            )

        # Resolve path to absolute
        resolved_path = self._resolve_path(input.get("path"))

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
                    "Use 'list_files' command to verify the file exists",
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
• Use 'read_file' command to copy precise text including all whitespace and formatting""",
                code="TEXT_NOT_FOUND",
                path=str(resolved_path),
                suggestions=[
                    "Check for typos or whitespace differences",
                    "Check for extra character escaping (content must exactly match the file)",
                    "Use 'read_file' command to copy precise text including all whitespace and formatting",
                ],
            )

        if len(parts) > 2:
            raise FileSystemError(
                message=f"""Multiple matches found: '{old_str}' appears {len(parts) - 1} times in '{resolved_path}'

str_replace requires exactly one occurrence. To fix:
• Add more surrounding context to old_str to make it unique
• Include 3+ lines before and after the target text
• Use 'create' command to create a new file or 'edit' to replace entire content""",
                code="MULTIPLE_MATCHES",
                path=str(resolved_path),
                suggestions=[
                    "Add more surrounding context to old_str to make it unique",
                    "Include 3+ lines before and after the target text",
                    "Use 'create' command to create a new file or 'edit' to replace entire content",
                ],
            )

        # Perform the replacement using join
        new_content = input.get("new_str").join(parts)

        # Write the new content back to the file
        resolved_path.write_text(new_content, encoding="utf-8")

        return f"File '{resolved_path}' modified successfully."

    async def _arun(self, input: dict[str, Any]) -> str:
        """Asynchronous wrapper for str_replace operation.

        Since file operations are synchronous, we can call _run directly.

        Args:
            input: Dictionary containing 'path', 'old_str', and 'new_str' keys.

        Returns:
            Success message or error message describing the result.
        """
        import asyncio

        return await asyncio.to_thread(lambda: self._run(input))

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
            return (self._get_effective_workdir() / path).resolve()

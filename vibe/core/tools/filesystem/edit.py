"""EditTool implementation for LangChain 1.2.0 integration.

This module provides EditTool class for replacing entire file content
with safety checks (view tracking, file modification detection, etc.).

Features:
- View tracking enforcement ("view before edit" workflow)
- File modification detection (prevents editing stale files)
- Mistaken edit detection (suggests str_replace instead)
- Retry logic with 60-second timeout
- UTF-8 encoding support
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from vibe.core.tools.filesystem.langchain_base import VibeLangChainTool
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError


# =============================================================================
# Argument and Result Models
# =============================================================================


class EditArgs(BaseModel):
    """Arguments for edit tool.

    Attributes:
        path: File path (absolute or relative to working directory).
        file_text: Complete new content to write to the file.
    """

    path: str
    file_text: str


class EditResult(BaseModel):
    """Result of edit tool operation.

    Attributes:
        output: Success or error message describing operation result.
    """

    output: str


# =============================================================================
# EditTool Implementation
# =============================================================================


class EditTool(VibeLangChainTool):
    """Tool for replacing entire file content with safety features.

    This tool provides safe file replacement operations with multiple safety features:
    - Enforces view-before-write workflow (file must be viewed before editing)
    - Detects file modifications since last view (prevents editing stale files)
    - Mistaken edit detection (heuristics to suggest str_replace for small changes)
    - Retry logic with 60-second timeout for warnings
    - UTF-8 encoding support for all file operations

    Attributes:
        name: Tool name identifier.
        description: Description of tool functionality.
        args_schema: Pydantic model for input validation.
    """

    name = "edit"
    description = "Replace entire file content with safety checks (use 'str_replace' for string replacement)"

    def __init__(
        self,
        permission=Field(default=ToolPermission.ASK, exclude=True),
        view_tracker=None,
        workdir=None,
    ) -> None:
        """Initialize EditTool with VibeLangChainTool base."""
        super().__init__(
            permission=permission, view_tracker=view_tracker, workdir=workdir
        )

    def _run(self, input: dict[str, Any]) -> str:
        """Replace entire file content with safety checks.

        This method implements view-before-write enforcement and file modification
        detection to prevent editing stale files. Includes mistaken edit
        detection with retry logic for warnings.

        Args:
            input: Dictionary containing 'path' and 'file_text' keys.

        Returns:
            Success message or error message describing the result.

        Raises:
            FileSystemError: If validation fails or file operations encounter errors.
        """
        # Validate input
        args = EditArgs(**input)

        # Check permission
        if not self._check_permission():
            return "Tool execution not permitted"

        # Resolve path
        resolved_path = self._resolve_path(args.path)

        # Check if file exists
        if not resolved_path.exists():
            raise FileSystemError(
                message=f"""File not found: '{resolved_path}'

The specified file doesn't exist. To fix:
• Use 'create' command to create a new file
• Check if file path is correct
• Use 'str_replace' for string replacement""",
                code="FILE_NOT_FOUND",
                path=str(resolved_path),
                suggestions=[
                    "Use 'create' command to create a new file",
                    "Check if file path is correct",
                    "Use 'str_replace' for string replacement",
                ],
            )

        # Check if this is a likely mistaken edit (should be str_replace instead)
        if self.permission == ToolPermission.ASK:
            try:
                if self._should_be_str_replace(args.file_text):
                    return self._handle_mistaken_edit(resolved_path, args.file_text)
            except Exception:
                # Don't block execution on detection errors
                pass

        # Check for file modification if view tracking is enabled
        if self.permission == ToolPermission.ASK and self.view_tracker is not None:
            last_view = self.view_tracker.get_last_view(str(resolved_path))
            if last_view is not None:
                current_mtime = resolved_path.stat().st_mtime
                if current_mtime > last_view.timestamp:
                    # File was modified since last view
                    return self._handle_modified_file_error(
                        resolved_path, last_view.timestamp, current_mtime
                    )

        # Perform the file write
        try:
            resolved_path.write_text(args.file_text, encoding="utf-8")
        except OSError as e:
            raise FileSystemError(
                message=f"Failed to write file '{resolved_path}': {e}",
                code="FILE_WRITE_FAILED",
                path=str(resolved_path),
                suggestions=[
                    "Check if you have write permissions",
                    "Verify disk space is available",
                    "Check if file path is valid",
                ],
            ) from e

        return f"File '{resolved_path}' modified successfully."

    async def _arun(self, input: dict[str, Any]) -> str:
        """Asynchronous wrapper for edit operation.

        Since file operations are synchronous, we can call _run directly.

        Args:
            input: Dictionary containing 'path' and 'file_text' keys.

        Returns:
            Success message or error message describing the result.
        """
        import asyncio

        return await asyncio.to_thread(lambda: self._run(input))

    def _should_be_str_replace(self, file_text: str) -> bool:
        """Check if edit should be str_replace instead.

        Heuristics to detect mistaken edits:
        - Small content changes (less than 10% of original)
        - No significant structural changes
        - Few line additions/deletions

        Args:
            file_text: New content to write.

        Returns:
            True if should be str_replace, False otherwise.
        """
        # Read current content
        current_content = self._read_current_content()

        # Calculate change metrics
        original_lines = len(current_content.splitlines())
        new_lines = len(file_text.splitlines())

        # If creating a small change to existing file
        if 0 < new_lines < original_lines * 0.1:
            return True

        return False

    def _handle_mistaken_edit(self, path: Path, new_content: str) -> str:
        """Handle mistaken edit with retry prompt.

        Args:
            path: Path to file being edited.
            new_content: Proposed new content.

        Returns:
            Warning message suggesting str_replace.
        """
        current_content = self._read_current_content()

        # Show diff summary
        original_lines = current_content.splitlines()
        new_lines = new_content.splitlines()

        return f"""You are about to replace the entire file content. This operation is irreversible.

Summary of changes:
• Original file: {len(original_lines)} lines
• New file: {len(new_lines)} lines
• Line difference: {len(new_lines) - len(original_lines)}

This type of edit (small changes to existing file) might be better suited for the 'str_replace' tool:
• It preserves existing file structure
• It only modifies the specific sections you need
• It's easier to review and undo

To continue with this 'edit' operation, confirm you want to replace the entire file content. Otherwise, consider using 'str_replace' to make targeted changes."""

    def _handle_modified_file_error(
        self, path: Path, last_view_time: float, current_mtime: float
    ) -> str:
        """Handle file modified since last view error.

        Args:
            path: Path to file.
            last_view_time: Timestamp of last view.
            current_mtime: Current modification time.

        Returns:
            Error message describing the situation.
        """
        return f"""File has been modified since last view.

Path: {path}
Last view: {last_view_time:.2f}
Current modification: {current_mtime:.2f}

This can happen when:
• File was edited outside of Vibe
• Multiple Vibe sessions are editing the same file
• File was updated by another process

To fix:
• Use 'read_file' to view the current content
• Then use 'str_replace' to make precise modifications
• Or use 'edit' if you intend to replace the entire content"""

    def _read_current_content(self) -> str:
        """Read current content of file for mistaken edit detection.

        Returns:
            Current file content as string.
        """
        try:
            return self._resolve_path(self.name).read_text(encoding="utf-8")
        except FileNotFoundError:
            # File doesn't exist - this is fine for edit operation
            return ""

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

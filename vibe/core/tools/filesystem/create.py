"""CreateTool implementation for creating new files with strict safety guarantees.

This module provides the CreateTool class that enables safe file creation operations.
The tool is designed to be a strict, safe operation that only creates NEW files,
matching TypeScript `FileEditor.create()` behavior exactly.

Features:
- Strict file creation (only creates NEW files)
- No overwrite option (matches TypeScript exactly)
- Automatic parent directory creation
- UTF-8 encoding for all file operations
- Optional view tracking for integration with edit workflows
- Helpful error messages guiding users to alternative tools

Example:
    ```python
    from vibe.core.tools.filesystem.create import CreateTool

    tool = CreateTool(workdir=Path("/project"))
    result = await tool.arun(path="test.py", file_text="print('hello')")
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from vibe.core.tools.base import BaseTool
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError

# =============================================================================
# Argument and Result Models
# =============================================================================


class CreateArgs(BaseModel):
    """Arguments for the create tool.

    Attributes:
        path: File path (absolute or relative to working directory).
        file_text: Content to write to the file.
    """

    path: str
    file_text: str


class CreateResult(BaseModel):
    """Result of the create tool operation.

    Attributes:
        output: Success or error message describing the operation result.
    """

    output: str


# =============================================================================
# CreateTool Implementation
# =============================================================================


class CreateTool(BaseTool):
    """Tool for creating new files with strict safety guarantees.

    This tool provides safe file creation operations with the following characteristics:
    - Only creates NEW files (fails if file already exists)
    - No overwrite option (intentional, matches TypeScript exactly)
    - Creates parent directories automatically if they don't exist
    - Writes files with UTF-8 encoding
    - Optional view tracking for integration with edit workflows

    The tool is simpler than WriteFileTool because it:
    - Has no view tracking requirement (unlike edit operations)
    - Has no modification detection (only creates new files)
    - Has no mistaken edit detection (not applicable for new files)
    - Has no retry logic (not applicable for new files)
    """

    def __init__(
        self,
        view_tracker: ViewTrackerService | None = None,
        workdir: Path | None = None,
    ) -> None:
        """Initialize CreateTool.

        Args:
            view_tracker: Optional service for tracking file views.
            workdir: Working directory for path resolution. Defaults to cwd if None.
        """
        super().__init__(
            name="create",
            description="Create new files (use 'edit' to replace entire file)",
            args_schema=CreateArgs,
        )
        self._view_tracker = view_tracker
        self._workdir = workdir or Path.cwd()

    async def _arun(
        self,
        path: str,
        file_text: str,
    ) -> CreateResult:
        """Execute the create tool operation.

        Creates a new file with the specified content. Fails if the file already
        exists, matching TypeScript FileEditor.create() behavior exactly.

        Args:
            path: File path (absolute or relative to working directory).
            file_text: Content to write to the file.

        Returns:
            CreateResult with success message.

        Raises:
            FileSystemError: If file already exists or file operations fail.
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(path)

        # Check if file already exists - STRICT check, no overwrite option
        if resolved_path.exists():
            raise FileSystemError(
                message=f"Cannot create file - it already exists: '{resolved_path}'\n\n• If you want to replace entire file: use 'edit' command\n• If you want to modify specific parts: use 'str_replace' command\n• If you want a different file: choose a different filename/location",
                code="FILE_ALREADY_EXISTS",
                path=str(resolved_path),
            )

        # Create parent directories if they don't exist
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file with UTF-8 encoding
        resolved_path.write_text(file_text, encoding="utf-8")

        # Record view after successful creation (optional, for edit workflow)
        if self._view_tracker is not None:
            self._view_tracker.record_view(str(resolved_path))

        return CreateResult(output=f"File '{resolved_path}' created successfully")

    def _run(self, **kwargs: Any) -> str:
        """Synchronous execution not supported."""
        raise NotImplementedError("CreateTool only supports async execution")

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

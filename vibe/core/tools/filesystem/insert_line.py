"""InsertLineTool implementation for inserting content at specific line numbers in files.

This module provides the InsertLineTool class that enables precise content insertion
at specific line positions in files with proper bounds validation and helpful error messages.

Features:
- 1-based line numbering (matching TypeScript behavior)
- Line bounds validation with helpful error messages
- UTF-8 encoding support
- Edit history tracking for undo functionality
- View tracking integration (optional)

Example:
    ```python
    from vibe.core.tools.filesystem.insert_line import InsertLineTool, InsertLineArgs

    tool = InsertLineTool(config=tool_config, state=tool_state)
    result = await tool.run(InsertLineArgs(path="test.py", new_str="new line", insert_line=3))
    ```
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.base import BaseTool, BaseToolConfig, BaseToolState
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError

# =============================================================================
# Argument and Result Models
# =============================================================================


class InsertLineArgs(BaseModel):
    """Arguments for the insert_line tool.

    Attributes:
        path: File path (absolute or relative to working directory).
        new_str: Content to insert at the specified line.
        insert_line: Line number where content should be inserted (1-based).
    """

    path: str
    new_str: str
    insert_line: int = Field(ge=1, description="Line number (1-based)")


class InsertLineResult(BaseModel):
    """Result of the insert_line tool operation.

    Attributes:
        output: Success or error message describing the operation result.
    """

    output: str


# =============================================================================
# Tool Configuration and State
# =============================================================================


class InsertLineToolConfig(BaseToolConfig):
    """Configuration for InsertLineTool.

    Extends BaseToolConfig to include the ViewTrackerService dependency.

    Attributes:
        view_tracker: Service for tracking file views during the session.
    """

    view_tracker: ViewTrackerService | None = None

    class Config:
        """Pydantic configuration for arbitrary types."""

        arbitrary_types_allowed = True


class InsertLineToolState(BaseToolState):
    """State for InsertLineTool.

    Tracks edit history for each file to support undo functionality.

    Attributes:
        edit_history: Dictionary mapping file paths to lists of previous content versions.
    """

    model_config = ConfigDict(extra="forbid")

    edit_history: dict[str, list[str]] = Field(default_factory=dict)


# =============================================================================
# InsertLineTool Implementation
# =============================================================================


class InsertLineTool(
    BaseTool[
        InsertLineArgs, InsertLineResult, InsertLineToolConfig, InsertLineToolState
    ]
):
    """Tool for inserting content at specific line numbers in files.

    This tool provides precise line insertion operations with several features:
    - 1-based line numbering (line 1 is the first line)
    - Insertion BEFORE the specified line number
    - Comprehensive bounds validation with helpful error messages
    - Edit history tracking for potential undo functionality
    - UTF-8 encoding for file operations
    - Integrates with ViewTrackerService for view tracking (optional)

    Attributes:
        name: Tool name identifier.
        description: Description of tool functionality.
        args_schema: Pydantic model for input validation.
        config: Tool configuration including ViewTrackerService.
        state: Tool state for tracking edit history.
    """

    name = "insert_line"
    description = "Insert content at a specific line in a file"
    args_schema: type[InsertLineArgs] = InsertLineArgs

    async def run(self, args: InsertLineArgs) -> InsertLineResult:
        """Execute the insert_line tool operation.

        Inserts the specified content at the given line number. The content is
        inserted before the existing line at that position.

        Args:
            args: InsertLineArgs containing path, new_str, and insert_line.

        Returns:
            InsertLineResult with success or error message.

        Raises:
            FileSystemError: If path resolution, file operations, or validation fails.
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(args.path)

        # Check if file exists
        if not resolved_path.exists():
            raise FileSystemError(
                message=f"""File not found: '{resolved_path}'

The specified file doesn't exist. Use 'write_file' command to create a new file, or check if the path is correct.""",
                code="FILE_NOT_FOUND",
                path=str(resolved_path),
                suggestions=[
                    "Use 'write_file' command to create a new file",
                    "Check the file path for typos or errors",
                    "Use 'list' command to verify the file exists",
                ],
            )

        # Read current content with UTF-8 encoding
        old_content = resolved_path.read_text(encoding="utf-8")

        # Determine new content based on file state
        if old_content == "":
            # Handle empty file case
            if args.insert_line == 1:
                new_content = args.new_str
            else:
                raise FileSystemError(
                    message=f"Invalid line number: {args.insert_line} is out of bounds for empty file '{resolved_path}'\n\nFor empty files, you can only insert at line 1. Use insertLine: 1 to add content to an empty file.",
                    code="LINE_OUT_OF_BOUNDS",
                    path=str(resolved_path),
                    suggestions=[
                        "For empty files, you can only insert at line 1",
                        "Use insertLine: 1 to add content to an empty file",
                        "Use 'write_file' command to add content to an empty file",
                    ],
                )
        else:
            # Handle non-empty file
            lines = old_content.split("\n")
            num_lines = len(lines)

            # Convert 1-based to 0-based index
            zero_index = args.insert_line - 1

            # Validate bounds: valid range is 0 to num_lines (inclusive)
            # This means insert_line can be from 1 to num_lines + 1
            if zero_index < 0 or zero_index > num_lines:
                raise FileSystemError(
                    message=f"Line number {args.insert_line} is out of bounds for '{resolved_path}' ({num_lines} lines)\n\nValid range: [1, {num_lines + 1}].\n• Use 1 to insert at the beginning\n• Use {num_lines + 1} to insert at the end\n• Use 'read_file' command first to see the file structure",
                    code="LINE_OUT_OF_BOUNDS",
                    path=str(resolved_path),
                    suggestions=[
                        "Use 1 to insert at the beginning",
                        f"Use {num_lines + 1} to insert at the end",
                        "Use 'read_file' command first to see the file structure",
                    ],
                )

            # Perform the insertion
            # Handle inserting at the end of a file with a trailing newline correctly
            if zero_index == num_lines and old_content.endswith("\n"):
                # This is an append operation on a file with a trailing newline
                # Insert before the empty string that represents the trailing newline
                # to avoid creating an extra blank line
                lines.insert(num_lines - 1, args.new_str)
            else:
                lines.insert(zero_index, args.new_str)

            new_content = "\n".join(lines)

        # Common success path: save to history, write file, update view tracker
        self._push_history(str(resolved_path), old_content)
        resolved_path.write_text(new_content, encoding="utf-8")

        if self.config.view_tracker is not None:
            self.config.view_tracker.record_view(str(resolved_path))

        return InsertLineResult(
            output=f"Content inserted into '{resolved_path}' at line {args.insert_line}."
        )

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
            return (self.config.effective_workdir / path).resolve()

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
        history = self.state.edit_history.setdefault(file_path, [])
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
        history = self.state.edit_history.get(file_path)
        if not history:
            return None
        return history.pop()

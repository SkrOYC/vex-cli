"""WriteFileTool implementation for creating and editing files with safety features.

This module provides the WriteFileTool class that enables safe file creation and
editing operations with built-in view tracking enforcement, file modification
detection, and mistaken edit prevention.

Features:
- Create new files or replace entire file content
- View tracking enforcement ("view before edit" workflow)
- File modification detection (prevents editing stale files)
- Mistaken edit detection (heuristic for str_replace detection)
- Retry logic with 60-second timeout for warnings
- Path resolution against working directory

Example:
    ```python
    from vibe.core.tools.filesystem.write_file import WriteFileTool, WriteFileArgs

    tool = WriteFileTool(config=tool_config, state=tool_state)
    result = await tool.run(WriteFileArgs(path="test.py", file_text="print('hello')"))
    ```
"""

from __future__ import annotations

from pathlib import Path
import time

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.base import BaseTool, BaseToolConfig, BaseToolState
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import (
    FILE_EXISTS_ERROR_TIMEOUT_MS,
    LENGTH_DIFF_RATIO_THRESHOLD,
    LINE_SIMILARITY_THRESHOLD,
    MIN_NEW_CONTENT_LENGTH,
    MIN_OLD_CONTENT_LENGTH,
    MISTAKEN_EDIT_TIMEOUT_MS,
    STR_REPLACE_LENGTH_RATIO_THRESHOLD,
    FileSystemError,
)

# =============================================================================
# Argument and Result Models
# =============================================================================


class WriteFileArgs(BaseModel):
    """Arguments for the write_file tool.

    Attributes:
        path: File path (absolute or relative to working directory).
        file_text: Content to write to the file.
    """

    path: str
    file_text: str


class WriteFileResult(BaseModel):
    """Result of the write_file tool operation.

    Attributes:
        output: Success or error message describing the operation result.
    """

    output: str


# =============================================================================
# Tool Configuration and State
# =============================================================================


class WriteFileToolConfig(BaseToolConfig):
    """Configuration for WriteFileTool.

    Extends BaseToolConfig to include the ViewTrackerService dependency.

    Attributes:
        view_tracker: Service for tracking file views during the session.
    """

    view_tracker: ViewTrackerService | None = None

    class Config:
        """Pydantic configuration for arbitrary types."""

        arbitrary_types_allowed = True


class WriteFileToolState(BaseToolState):
    """State for WriteFileTool.

    Tracks warned operations and file exists error flow for retry logic.

    Attributes:
        warned_operations: Tracks warned edit operations by file path.
        file_exists_operations: Tracks file exists error flow by file path.
    """

    model_config = ConfigDict(extra="forbid")

    warned_operations: dict[str, dict[str, int | str]] = Field(default_factory=dict)
    file_exists_operations: dict[str, dict[str, int]] = Field(default_factory=dict)


# =============================================================================
# WriteFileTool Implementation
# =============================================================================


class WriteFileTool(
    BaseTool[WriteFileArgs, WriteFileResult, WriteFileToolConfig, WriteFileToolState]
):
    """Tool for creating new files and replacing entire file content.

    This tool provides safe file writing operations with multiple safety features:
    - Enforces "view before edit" workflow using ViewTrackerService
    - Detects file modifications since last view
    - Prevents mistaken edit commands that should be str_replace
    - Allows retry within 60 seconds of warnings

    The tool determines operation type based on file existence:
    - Create: File does not exist, creates new file
    - Edit: File exists, replaces entire content

    Attributes:
        name: Tool name identifier.
        description: Description of tool functionality.
        args_schema: Pydantic model for input validation.
        config: Tool configuration including ViewTrackerService.
        state: Tool state for tracking retry operations.
    """

    name = "write_file"
    description = "Create new files or replace entire file content. Use 'create' for new files and 'edit' for replacing entire file content. Enforces viewing files before editing."
    args_schema: type[WriteFileArgs] = WriteFileArgs

    # Constants matching TypeScript implementation
    _HASH_MULTIPLIER: int = 31
    _BIT_MASK_32: int = 0x1_00_00_00_00  # 2^32
    _BASE_36: int = 36

    async def run(self, args: WriteFileArgs) -> WriteFileResult:
        """Execute the write_file tool operation.

        Determines whether to create a new file or edit an existing file based on
        whether the file already exists.

        Args:
            args: WriteFileArgs containing path and file_text.

        Returns:
            WriteFileResult with success or error message.

        Raises:
            FileSystemError: If path resolution or file operations fail.
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(args.path)

        # Check if file exists to determine operation type
        file_exists = resolved_path.exists()

        if file_exists:
            # Edit operation - replace entire file content
            return await self._perform_edit(resolved_path, args.file_text)
        else:
            # Create operation - create new file
            return await self._perform_create(resolved_path, args.file_text)

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
    # Create Operation
    # =============================================================================

    async def _perform_create(self, path: Path, file_text: str) -> WriteFileResult:
        """Create a new file with the given content.

        Args:
            path: Absolute path to the new file.
            file_text: Content to write to the file.

        Returns:
            WriteFileResult with success message.

        Raises:
            FileSystemError: If file already exists or directory creation fails.
        """
        # Check if file already exists
        if path.exists():
            raise FileSystemError(
                message=f"Cannot create file - it already exists: '{path}'",
                code="FILE_EXISTS",
                path=str(path),
                suggestions=[
                    "Use 'edit' command to replace entire file content",
                    "Use 'str_replace' command to modify specific parts",
                    "Choose a different filename or location",
                ],
            )

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file with UTF-8 encoding
        path.write_text(file_text, encoding="utf-8")

        # Record view after successful creation
        if self.config.view_tracker is not None:
            self.config.view_tracker.record_view(str(path))

        return WriteFileResult(output=f"File '{path}' created successfully.")

    # =============================================================================
    # Edit Operation
    # =============================================================================

    async def _perform_edit(self, path: Path, file_text: str) -> WriteFileResult:
        """Edit an existing file, replacing its entire content.

        Args:
            path: Absolute path to the file.
            file_text: New content to write.

        Returns:
            WriteFileResult with success message.

        Raises:
            FileSystemError: If file not viewed, modified since view, or mistaken edit detected.
        """
        # Clean up expired warnings periodically
        self._cleanup_expired_warnings()

        # Check if file was viewed before editing
        if self.config.view_tracker is None:
            raise FileSystemError(
                message="View tracker not configured",
                code="CONFIG_ERROR",
                path=str(path),
                suggestions=["Contact administrator to configure view tracking"],
            )

        if not self.config.view_tracker.has_been_viewed(str(path)):
            raise FileSystemError(
                message=f"File '{path}' must be viewed before editing.",
                code="FILE_NOT_VIEWED",
                path=str(path),
                suggestions=[
                    "Use 'read_file' command to examine the file before editing"
                ],
            )

        # Check if file was modified after the last view
        last_view_timestamp = self.config.view_tracker.get_last_view_timestamp(
            str(path)
        )
        if last_view_timestamp is not None:
            try:
                last_modified = path.stat().st_mtime * 1000  # Convert to milliseconds
            except OSError:
                # If we can't get file stats, proceed with the edit
                last_modified = 0

            if last_modified > last_view_timestamp:
                raise FileSystemError(
                    message=f"File '{path}' has been modified since it was last viewed.",
                    code="FILE_MODIFIED",
                    path=str(path),
                    suggestions=[
                        "Use 'read_file' command to see current content before editing"
                    ],
                )

        # Read current content for mistaken edit detection
        old_content = path.read_text(encoding="utf-8")

        # Check if this is following guidance from a "File already exists" error
        if self._is_following_file_exists_guidance(str(path)):
            # This is following system guidance - allow the operation and clean up tracking
            self.state.file_exists_operations.pop(str(path), None)
        elif self._is_likely_str_replace(old_content, file_text):
            # Check if this is a retry of a previously warned operation
            if self._is_previously_warned_operation(str(path), old_content, file_text):
                # This is a retry - allow the operation and clean up the warning
                self.state.warned_operations.pop(str(path), None)
            else:
                # First time warning - track the operation and throw the error
                self._track_warned_operation(str(path), old_content, file_text)
                raise FileSystemError(
                    message="Likely mistaken usage of 'edit' command detected.",
                    code="MISTAKEN_EDIT",
                    path=str(path),
                    suggestions=[
                        "Consider using 'str_replace' command instead for targeted changes",
                        "For str_replace, provide exact text to replace (oldStr) and new text (newStr)",
                        "Include 3+ lines of context before and after the target text in oldStr",
                        "If you intended to use 'edit', try the same command again within 60 seconds",
                    ],
                )

        # Write new content with UTF-8 encoding
        path.write_text(file_text, encoding="utf-8")

        # Record view after successful edit
        self.config.view_tracker.record_view(str(path))

        return WriteFileResult(output=f"File '{path}' edited successfully.")

    # =============================================================================
    # Hash Function (matching TypeScript implementation)
    # =============================================================================

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate a simple hash of the content for tracking operations.

        Matches TypeScript implementation exactly:
        - HASH_MULTIPLIER = 31
        - BIT_MASK_32 = 0x1_00_00_00_00 (2^32)
        - BASE_36 for string representation

        Args:
            content: The content to hash.

        Returns:
            A string hash of the content.
        """
        hash_value = 0
        for i in range(len(content)):
            char = content[i]
            hash_value = (
                hash_value * WriteFileTool._HASH_MULTIPLIER + ord(char)
            ) % WriteFileTool._BIT_MASK_32

        return str(abs(hash_value)).zfill(WriteFileTool._BASE_36)

    # =============================================================================
    # Mistaken Edit Detection
    # =============================================================================

    def _is_likely_str_replace(self, old_content: str, new_content: str) -> bool:
        """Check if this edit might be a mistaken str_replace command.

        Detects when an 'edit' command should likely be 'str_replace' based on:
        1. New content is significantly smaller than old content
        2. High line similarity between old and new content

        Args:
            old_content: Current content of the file.
            new_content: Proposed new content.

        Returns:
            True if this appears to be a targeted replacement rather than full rewrite.
        """
        # If either content is empty, it's not a targeted replacement
        if not old_content or not new_content:
            return False

        # If the new content is significantly smaller, it might be a targeted replacement
        length_ratio = len(new_content) / len(old_content)
        if length_ratio < STR_REPLACE_LENGTH_RATIO_THRESHOLD:
            return True

        # Check if content suggests targeted replacement rather than full rewrite
        return self._has_targeted_replacement_characteristics(old_content, new_content)

    def _has_targeted_replacement_characteristics(
        self, old_content: str, new_content: str
    ) -> bool:
        """Check if content has characteristics of a targeted replacement.

        For longer files, checks line similarity and length difference ratio.

        Args:
            old_content: Current content of the file.
            new_content: Proposed new content.

        Returns:
            True if content suggests targeted replacement.
        """
        # For longer files, check line similarity
        if (
            len(old_content) > MIN_OLD_CONTENT_LENGTH
            and len(new_content) > MIN_NEW_CONTENT_LENGTH
        ):
            similarity = self._calculate_line_similarity(old_content, new_content)
            if similarity > LINE_SIMILARITY_THRESHOLD:
                return True

        # Check if line counts are similar (suggesting targeted changes)
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        if old_lines and new_lines:
            length_diff_ratio = abs(len(old_lines) - len(new_lines)) / max(
                len(old_lines), len(new_lines)
            )

            if length_diff_ratio < LENGTH_DIFF_RATIO_THRESHOLD:
                similarity = self._calculate_line_similarity(old_content, new_content)
                return similarity > LINE_SIMILARITY_THRESHOLD

        return False

    def _calculate_line_similarity(self, old_content: str, new_content: str) -> float:
        """Calculate similarity between old and new content based on matching lines.

        Args:
            old_content: Current content of the file.
            new_content: Proposed new content.

        Returns:
            Similarity ratio between 0 and 1.
        """
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")
        old_lines_set = set(old_lines)

        matching_lines = sum(1 for line in new_lines if line in old_lines_set)

        return matching_lines / max(len(old_lines), len(new_lines))

    # =============================================================================
    # Retry Tracking
    # =============================================================================

    def _is_following_file_exists_guidance(self, file_path: str) -> bool:
        """Check if this edit is following guidance from a "File already exists" error.

        Args:
            file_path: The path to the file being edited.

        Returns:
            True if this operation is following system guidance and within timeout.
        """
        tracking = self.state.file_exists_operations.get(file_path)

        if not tracking:
            return False

        # Check if it's within timeout
        timestamp = tracking.get("timestamp")
        is_within_timeout = (
            isinstance(timestamp, int)
            and int(time.time() * 1000) - timestamp < FILE_EXISTS_ERROR_TIMEOUT_MS
        )

        return is_within_timeout

    def _is_previously_warned_operation(
        self, file_path: str, old_content: str, new_content: str
    ) -> bool:
        """Check if this edit has been previously warned about.

        Args:
            file_path: The path to the file being edited.
            old_content: Current content of the file.
            new_content: Proposed new content.

        Returns:
            True if this operation was previously warned about and within timeout.
        """
        warning = self.state.warned_operations.get(file_path)

        if not warning:
            return False

        # Check if it's the same operation (same file and content hashes)
        is_same_operation = warning.get("old_content_hash") == self._hash_content(
            old_content
        ) and warning.get("new_content_hash") == self._hash_content(new_content)

        # Check timeout
        warning_timestamp = warning.get("timestamp")
        is_within_timeout = (
            isinstance(warning_timestamp, int)
            and int(time.time() * 1000) - warning_timestamp < MISTAKEN_EDIT_TIMEOUT_MS
        )

        return is_same_operation and is_within_timeout

    def _track_warned_operation(
        self, file_path: str, old_content: str, new_content: str
    ) -> None:
        """Track a warned edit operation.

        Args:
            file_path: The path to the file being edited.
            old_content: Current content of the file.
            new_content: Proposed new content.
        """
        self.state.warned_operations[file_path] = {
            "timestamp": int(time.time() * 1000),
            "old_content_hash": self._hash_content(old_content),
            "new_content_hash": self._hash_content(new_content),
        }

    def _cleanup_expired_warnings(self) -> None:
        """Clean up expired warned operations and file exists error tracking."""
        current_time = int(time.time() * 1000)

        # Clean up expired warned operations
        expired_warned = []
        for key, warning in self.state.warned_operations.items():
            warning_timestamp = warning.get("timestamp")
            if (
                isinstance(warning_timestamp, int)
                and current_time - warning_timestamp >= MISTAKEN_EDIT_TIMEOUT_MS
            ):
                expired_warned.append(key)
        for key in expired_warned:
            self.state.warned_operations.pop(key, None)

        # Clean up expired file exists error tracking
        expired_file_exists = []
        for key, tracking in self.state.file_exists_operations.items():
            tracking_timestamp = tracking.get("timestamp")
            if (
                isinstance(tracking_timestamp, int)
                and current_time - tracking_timestamp >= FILE_EXISTS_ERROR_TIMEOUT_MS
            ):
                expired_file_exists.append(key)
        for key in expired_file_exists:
            self.state.file_exists_operations.pop(key, None)

"""EditTool implementation for replacing entire file content with safety features.

This module provides the EditTool class that enables safe file editing operations
with comprehensive safety checks. The tool matches TypeScript `FileEditor.edit()`
behavior exactly.

Features:
- View tracking enforcement ("view before edit" workflow)
- File modification detection (prevents editing stale files)
- Mistaken edit detection (heuristics to suggest edit_file instead)
- Retry logic with 60-second timeout for warnings
- UTF-8 encoding for all file operations
- Path resolution against working directory

Example:
    ```python
    from vibe.core.tools.filesystem.edit import EditTool, EditArgs

    tool = EditTool(config=tool_config, state=tool_state)
    result = await tool.run(EditArgs(path="test.py", file_text="print('hello')"))
    ```
"""

from __future__ import annotations

from pathlib import Path
import time

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.base import BaseTool, BaseToolConfig, BaseToolState
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import (
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


class EditArgs(BaseModel):
    """Arguments for the edit tool.

    Attributes:
        path: File path (absolute or relative to working directory).
        file_text: New entire content to write to the file.
    """

    path: str
    file_text: str


class EditResult(BaseModel):
    """Result of the edit tool operation.

    Attributes:
        output: Success or error message describing the operation result.
    """

    output: str


class WarnedOperation(BaseModel):
    """Represents a warned edit operation for retry logic.

    Attributes:
        timestamp: When the warning was issued (milliseconds since epoch).
        old_content_hash: Hash of the old content for comparison.
        new_content_hash: Hash of the new content for comparison.
    """

    timestamp: int
    old_content_hash: str
    new_content_hash: str

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Tool Configuration and State
# =============================================================================


class EditToolConfig(BaseToolConfig):
    """Configuration for EditTool.

    Extends BaseToolConfig to include the optional ViewTrackerService dependency.

    Attributes:
        view_tracker: Optional service for tracking file views during the session.
    """

    view_tracker: ViewTrackerService | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EditToolState(BaseToolState):
    """State for EditTool.

    Tracks warned operations for retry logic.

    Attributes:
        warned_operations: Dictionary tracking warned edit operations by file path.
    """

    model_config = ConfigDict(extra="forbid")

    warned_operations: dict[str, WarnedOperation] = Field(default_factory=dict)


# =============================================================================
# EditTool Implementation
# =============================================================================


class EditTool(BaseTool[EditArgs, EditResult, EditToolConfig, EditToolState]):
    """Tool for replacing entire file content with comprehensive safety checks.

    This tool provides safe file editing operations with multiple safety features:
    - Enforces "view before edit" workflow using ViewTrackerService
    - Detects file modifications since last view
    - Prevents mistaken edit commands that should be edit_file (str_replace)
    - Allows retry within 60 seconds of warnings

    The tool is designed for full file content replacement. For targeted string
    replacement, use EditFileTool instead.

    Attributes:
        name: Tool name identifier.
        description: Description of tool functionality.
        args_schema: Pydantic model for input validation.
    """

    name = "edit"
    description = (
        "Replace entire file content with safety checks "
        "(use 'edit_file' for str_replace)"
    )
    args_schema: type[EditArgs] = EditArgs

    # Constants matching TypeScript implementation
    _HASH_MULTIPLIER: int = 31
    _BIT_MASK_32: int = 0x1_00_00_00_00  # 2^32
    _BASE_36: int = 36

    async def run(self, args: EditArgs) -> EditResult:
        """Execute the edit tool operation.

        Replaces the entire content of an existing file with the provided content.
        Enforces view tracking, checks for file modifications, and detects
        mistaken edit commands.

        Args:
            args: EditArgs containing path and file_text.

        Returns:
            EditResult with success or error message.

        Raises:
            FileSystemError: If path resolution, file operations, or validation fails.
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(args.path)

        # Clean up expired warnings periodically
        self._cleanup_expired_warnings()

        # Check if file exists
        if not resolved_path.exists():
            raise FileSystemError(
                message=f"File not found: '{resolved_path}'\n\n"
                "The specified file doesn't exist. "
                "Use 'create' command to create a new file, or check the file path.",
                code="FILE_NOT_FOUND",
                path=str(resolved_path),
            )

        # Check if file was viewed before editing
        if (
            self.config.view_tracker is None
            or not self.config.view_tracker.has_been_viewed(str(resolved_path))
        ):
            raise FileSystemError(
                message=f"File '{resolved_path}' must be viewed before editing\n\n"
                "Use 'read_file' command to examine file content first.",
                code="FILE_NOT_VIEWED",
                path=str(resolved_path),
                suggestions=["Use 'read_file' command to examine file content first."],
            )
            raise FileSystemError(
                message=f"File '{resolved_path}' must be viewed before editing\n\n"
                "Use 'read_file' command to examine file content first.",
                code="FILE_NOT_VIEWED",
                path=str(resolved_path),
                suggestions=["Use 'read_file' command to examine file content first."],
            )

        # Check if file was modified after the last view
        last_view_timestamp = self.config.view_tracker.get_last_view_timestamp(
            str(resolved_path)
        )
        if last_view_timestamp is not None:
            try:
                last_modified = (
                    resolved_path.stat().st_mtime * 1000
                )  # Convert to milliseconds
            except OSError:
                # If we can't get file stats, proceed with the edit
                last_modified = 0

            if last_modified > last_view_timestamp:
                raise FileSystemError(
                    message=f"File '{resolved_path}' has been modified since it was last viewed\n\n"
                    "Use 'read_file' command to see current content.",
                    code="FILE_MODIFIED",
                    path=str(resolved_path),
                    suggestions=["Use 'read_file' command to see current content."],
                )

        # Read current content for mistaken edit detection
        old_content = resolved_path.read_text(encoding="utf-8")

        # Check if this is a likely mistaken edit (should be edit_file instead)
        if self._is_likely_mistaken_edit(
            old_content, args.file_text, str(resolved_path)
        ):
            old_content_hash = self._hash_content(old_content)
            new_content_hash = self._hash_content(args.file_text)

            # Check if this is a retry of a previously warned operation
            if self._check_for_retry(
                str(resolved_path), old_content_hash, new_content_hash
            ):
                # This is a retry - allow the operation and clean up the warning
                self.state.warned_operations.pop(str(resolved_path), None)
            else:
                # First time warning - track the operation and throw the error
                self._record_warning(
                    str(resolved_path), old_content_hash, new_content_hash
                )
                raise FileSystemError(
                    message="Likely mistaken usage of 'edit' command detected\n\n"
                    "Consider using 'edit_file' (str_replace) instead:\n"
                    "• Include more context in the old_str parameter\n"
                    "• Use 3+ lines before and after the target text\n"
                    "• Try this command again to proceed (60-second timeout)",
                    code="MISTAKEN_EDIT",
                    path=str(resolved_path),
                    suggestions=[
                        "Consider using 'edit_file' (str_replace) instead",
                        "Include more context in the old_str parameter",
                        "Use 3+ lines before and after the target text",
                        "Try this command again to proceed (60-second timeout)",
                    ],
                )

        # Write new content with UTF-8 encoding
        resolved_path.write_text(args.file_text, encoding="utf-8")

        # Record view after successful edit
        self.config.view_tracker.record_view(str(resolved_path))

        return EditResult(output=f"File '{resolved_path}' modified successfully")

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
    # Hash Function (matching TypeScript implementation)
    # =============================================================================

    @staticmethod
    def _to_base36(n: int) -> str:
        """Converts a non-negative integer to its base-36 representation.

        Args:
            n: Non-negative integer to convert.

        Returns:
            Base-36 string representation (digits 0-9, letters a-z).
        """
        if n == 0:
            return "0"

        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        result = ""
        while n > 0:
            n, remainder = divmod(n, 36)
            result = chars[remainder] + result
        return result

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate a simple hash of the content for tracking operations.

        Matches TypeScript implementation exactly:
        - HASH_MULTIPLIER = 31
        - BIT_MASK_32 = 0x1_00_00_00_00 (2^32)
        - BASE_36 for string representation (using toString(36))

        Args:
            content: The content to hash.

        Returns:
            A base-36 string hash of the content (variable length).
        """
        hash_value = 0
        for char in content:
            hash_value = (
                hash_value * EditTool._HASH_MULTIPLIER + ord(char)
            ) % EditTool._BIT_MASK_32

        return EditTool._to_base36(abs(hash_value))

    # =============================================================================
    # Mistaken Edit Detection
    # =============================================================================

    def _is_likely_mistaken_edit(
        self, old_content: str, new_content: str, file_path: str
    ) -> bool:
        """Check if edit should be edit_file instead.

        Uses majority voting: at least 2 of 3 heuristics must trigger.

        Args:
            old_content: Current content of the file.
            new_content: Proposed new content.
            file_path: Path to the file being edited.

        Returns:
            True if this appears to be a targeted replacement rather than full rewrite.
        """
        # Skip if content too short for meaningful comparison
        if len(old_content) < MIN_OLD_CONTENT_LENGTH:
            return False
        if len(new_content) < MIN_NEW_CONTENT_LENGTH:
            return False

        # Run heuristics
        length_ratio_ok = self._check_length_ratio(old_content, new_content)
        similarity_ok = self._check_line_similarity(old_content, new_content)
        length_diff_ok = self._check_length_diff(old_content, new_content)

        # Majority voting (2 of 3)
        checks_passed = sum([length_ratio_ok, similarity_ok, length_diff_ok])
        majority_threshold = 2
        return checks_passed >= majority_threshold

    def _check_length_ratio(self, old_content: str, new_content: str) -> bool:
        """Check if new content is significantly smaller than old content.

        Args:
            old_content: Current content of the file.
            new_content: Proposed new content.

        Returns:
            True if new content is less than 30% of old content length.
        """
        length_ratio = len(new_content) / len(old_content)
        return length_ratio < STR_REPLACE_LENGTH_RATIO_THRESHOLD

    def _check_line_similarity(self, old_content: str, new_content: str) -> bool:
        """Check if old and new content have high line similarity.

        Uses Jaccard similarity (intersection / union) of unique lines for
        accurate comparison. This avoids skewing from duplicate lines.

        Args:
            old_content: Current content of the file.
            new_content: Proposed new content.

        Returns:
            True if more than 70% of unique lines match.
        """
        old_lines = set(old_content.split("\n"))
        new_lines = set(new_content.split("\n"))

        if len(old_lines) == 0 and len(new_lines) == 0:
            return True

        intersection_len = len(old_lines.intersection(new_lines))
        union_len = len(old_lines.union(new_lines))

        if union_len == 0:
            return True

        similarity = intersection_len / union_len
        return similarity > LINE_SIMILARITY_THRESHOLD

    def _check_length_diff(self, old_content: str, new_content: str) -> bool:
        """Check if line counts are similar between old and new content.

        Args:
            old_content: Current content of the file.
            new_content: Proposed new content.

        Returns:
            True if line count difference is less than 30%.
        """
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        line_diff = abs(len(old_lines) - len(new_lines))
        length_diff_ratio = line_diff / len(old_lines)

        return length_diff_ratio < LENGTH_DIFF_RATIO_THRESHOLD

    # =============================================================================
    # Retry Tracking
    # =============================================================================

    def _check_for_retry(
        self, file_path: str, old_content_hash: str, new_content_hash: str
    ) -> bool:
        """Check if this is a retry of previously warned operation.

        Returns True if warned within 60 seconds and hashes match.

        Args:
            file_path: The path to the file being edited.
            old_content_hash: Hash of the old content.
            new_content_hash: Hash of the new content.

        Returns:
            True if this is a valid retry, False otherwise.
        """
        if file_path not in self.state.warned_operations:
            return False

        warning = self.state.warned_operations[file_path]
        current_time = int(time.time() * 1000)
        time_diff = current_time - warning.timestamp

        if time_diff > MISTAKEN_EDIT_TIMEOUT_MS:
            # Expired warning, remove it
            del self.state.warned_operations[file_path]
            return False

        # Check if hashes match (same operation)
        hash_matches = (
            warning.old_content_hash == old_content_hash
            and warning.new_content_hash == new_content_hash
        )

        return hash_matches

    def _record_warning(
        self, file_path: str, old_content_hash: str, new_content_hash: str
    ) -> None:
        """Record a mistaken edit warning for retry tracking.

        Args:
            file_path: The path to the file being edited.
            old_content_hash: Hash of the old content.
            new_content_hash: Hash of the new content.
        """
        self.state.warned_operations[file_path] = WarnedOperation(
            timestamp=int(time.time() * 1000),
            old_content_hash=old_content_hash,
            new_content_hash=new_content_hash,
        )

    def _cleanup_expired_warnings(self) -> None:
        """Clean up expired warned operations."""
        current_time = int(time.time() * 1000)

        expired_keys = []
        for key, warning in self.state.warned_operations.items():
            if current_time - warning.timestamp >= MISTAKEN_EDIT_TIMEOUT_MS:
                expired_keys.append(key)

        for key in expired_keys:
            self.state.warned_operations.pop(key, None)

"""ListFilesTool implementation for discovering files and directories with glob patterns.

This module provides the ListFilesTool class that enables file and directory discovery
with glob pattern matching, recursive traversal, and file metadata.

Features:
- find_files command: Glob-based file discovery with pattern matching
- list command: Directory listing with metadata and tree structure
- Pattern matching: Support for *, **, ?, and [abc] wildcards
- Exclude patterns: Filter out files matching certain patterns
- Recursive search: Search subdirectories with max depth control
- File metadata: Line counts, file sizes in human-readable format
- Output truncation: Limit output at 80,000 characters

Example:
    ```python
    from vibe.core.tools.filesystem.list_files import ListFilesTool, ListFilesArgs

    tool = ListFilesTool(config=tool_config, state=tool_state)
    result = await tool.run(ListFilesArgs(path=".", patterns=["*.py"]))
    ```
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vibe.core.tools.base import BaseTool
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import OUTPUT_LIMIT, FileSystemError

# =============================================================================
# Argument and Result Models
# =============================================================================


class ListFilesArgs(BaseModel):
    """Arguments for the list_files tool.

    Attributes:
        path: Directory path (absolute or relative to working directory).
        patterns: List of glob patterns to match files against.
        exclude: List of patterns to exclude matching files.
        recursive: Whether to search subdirectories (default: True).
        max_results: Maximum number of files to return (default: 1000, range: 1-10000).
        include_hidden: Whether to include hidden files and directories (default: False).
        max_depth: Maximum depth for recursive listing (default: 3, range: 1-10).
    """

    path: str = "."
    patterns: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    recursive: bool = True
    max_results: int = Field(default=1000, ge=1, le=10_000)
    include_hidden: bool = False
    max_depth: int = Field(default=3, ge=1, le=10)


class ListFilesResult(BaseModel):
    """Result of the list_files tool operation.

    Attributes:
        output: Formatted file listing or directory contents.
    """

    output: str


# =============================================================================
# ListFilesTool Implementation
# =============================================================================


class ListFilesTool(BaseTool):
    """Tool for listing files and directories with pattern matching and metadata.

    This tool provides comprehensive file and directory discovery capabilities:
    - find_files: Glob-based file discovery with pattern matching
    - list: Directory listing with metadata and tree structure

    When patterns are provided, it operates in find_files mode.
    When no patterns are provided, it operates in list mode.
    """

    # Class attributes for compatibility (set via __init__ but accessible as class attrs)
    name: str = "list_files"
    description: str = "List files and directories with pattern matching and metadata. Use patterns for find_files mode, or list directory contents."

    # Constants
    _BYTE_SIZE_THRESHOLD: int = 1024  # Bytes per KB

    def __init__(
        self,
        view_tracker: ViewTrackerService | None = None,
        workdir: Path | None = None,
    ) -> None:
        """Initialize ListFilesTool.

        Args:
            view_tracker: Optional service for tracking file views.
            workdir: Working directory for path resolution. Defaults to cwd if None.
        """
        super().__init__(
            name="list_files",
            description="List files and directories with pattern matching and metadata. Use patterns for find_files mode, or list directory contents.",
            args_schema=ListFilesArgs,
        )
        self._view_tracker = view_tracker
        self._workdir = workdir or Path.cwd()

    def _run(self, **kwargs: Any) -> str:
        """Synchronous execution not supported."""
        raise NotImplementedError("ListFilesTool only supports async execution")

    async def _arun(
        self,
        path: str = ".",
        patterns: list[str] = [],
        exclude: list[str] = [],
        recursive: bool = True,
        max_results: int = 1000,
        include_hidden: bool = False,
        max_depth: int = 3,
    ) -> ListFilesResult:
        """Execute the list_files tool operation.

        Determines whether to use find_files or list mode based on whether
        patterns are provided.

        Args:
            path: Directory path (absolute or relative to working directory).
            patterns: List of glob patterns to match files against.
            exclude: List of patterns to exclude matching files.
            recursive: Whether to search subdirectories (default: True).
            max_results: Maximum number of files to return (default: 1000, range: 1-10000).
            include_hidden: Whether to include hidden files and directories (default: False).
            max_depth: Maximum depth for recursive listing (default: 3, range: 1-10).

        Returns:
            ListFilesResult with formatted output.

        Raises:
            FileSystemError: If path resolution or file operations fail.
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(path)

        # Validate path exists
        if not resolved_path.exists():
            raise FileSystemError(
                message=f"Directory not found: '{path}'",
                code="DIRECTORY_NOT_FOUND",
                path=str(resolved_path),
                suggestions=[
                    f"The specified directory doesn't exist in current working directory: {self._workdir}",
                    "Use 'list' command to explore available directories",
                    "Check if the path is correct",
                ],
            )

        # Check if path is a directory
        if not resolved_path.is_dir():
            raise FileSystemError(
                message=f"Path is not a directory: '{path}'",
                code="PATH_NOT_DIRECTORY",
                path=str(resolved_path),
                suggestions=[
                    "Use 'view' command to examine file contents",
                    "Specify a directory path to list",
                ],
            )

        # Determine mode based on whether patterns are provided
        if patterns:
            # find_files mode
            output = await self._find_files(
                resolved_path, patterns, exclude, recursive, max_results, include_hidden
            )
        else:
            # list mode
            output = await self._list(
                resolved_path, recursive, max_results, include_hidden, max_depth
            )

        # Apply output truncation
        output = self._truncate_output(output, patterns is not None)

        return ListFilesResult(output=output)

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
    # find_files Command
    # =============================================================================

    async def _find_files(
        self,
        root: Path,
        patterns: list[str],
        exclude: list[str],
        recursive: bool,
        max_results: int,
        include_hidden: bool,
    ) -> str:
        """Find files matching glob patterns.

        Args:
            root: Root directory to search in.
            patterns: List of glob patterns to match files against.
            exclude: List of patterns to exclude matching files.
            recursive: Whether to search subdirectories.
            max_results: Maximum number of files to return.
            include_hidden: Whether to include hidden files.

        Returns:
            Formatted list of matching files.
        """
        found_files: list[Path] = []

        # Process each pattern
        for pattern in patterns:
            # Use pathlib glob for matching
            # rglob handles ** automatically, glob handles non-recursive
            if recursive:
                matches = root.rglob(pattern)
            else:
                matches = root.glob(pattern)

            for file_path in matches:
                # Only process files (not directories)
                if not file_path.is_file():
                    continue

                # Check hidden file filter
                if not include_hidden and file_path.name.startswith("."):
                    continue

                # Check exclude patterns using relative path
                relative_path = str(file_path.relative_to(root))
                if self._matches_any_exclude(relative_path, exclude):
                    continue

                # Check max results
                if len(found_files) >= max_results:
                    break

                # Avoid duplicates
                if file_path not in found_files:
                    found_files.append(file_path)

        # Format output with relative paths
        if not found_files:
            return f"No files found matching patterns: {', '.join(patterns)}"

        # Sort files alphabetically (case-insensitive)
        found_files.sort(key=lambda p: str(p).casefold())

        # Create relative paths for output
        relative_files = [str(f.relative_to(root)) for f in found_files]

        file_list = "\n".join(relative_files)
        truncation_msg = (
            f" (limited to {max_results})" if len(found_files) >= max_results else ""
        )

        return f"Found {len(found_files)} files{truncation_msg}\n\n{file_list}"

    def _matches_any_exclude(self, file_path: str, exclude_patterns: list[str]) -> bool:
        """Check if file path matches any exclude pattern.

        Args:
            file_path: The file path to check (relative to search root).
            exclude_patterns: List of patterns to exclude.

        Returns:
            True if the file matches any exclude pattern.
        """
        for exclude_pattern in exclude_patterns:
            if fnmatch(file_path, exclude_pattern):
                return True
        return False

    # =============================================================================
    # list Command
    # =============================================================================

    async def _list(
        self,
        root: Path,
        recursive: bool,
        max_results: int,
        include_hidden: bool,
        max_depth: int,
    ) -> str:
        """List directory contents.

        Args:
            root: Root directory to list.
            recursive: Whether to search subdirectories.
            max_results: Maximum number of files to return.
            include_hidden: Whether to include hidden files.
            max_depth: Maximum depth for recursive listing.

        Returns:
            Formatted directory listing.
        """
        if recursive:
            return await self._list_recursive(root, max_depth, include_hidden)
        else:
            return self._list_single(root, include_hidden)

    def _list_single(self, root: Path, include_hidden: bool) -> str:
        """List directory contents (non-recursive).

        Args:
            root: Root directory to list.
            include_hidden: Whether to include hidden files.

        Returns:
            Formatted directory listing.
        """
        entries = list(root.iterdir())

        # Filter hidden files if needed
        if not include_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        # Sort: directories first, then files alphabetically (case-insensitive)
        entries.sort(key=lambda e: (not e.is_dir(), e.name.casefold()))

        # Build output
        lines: list[str] = []

        for entry in entries:
            if entry.is_dir():
                lines.append(f"  [DIR]  {entry.name}/")
            else:
                # Get file metadata
                line_count = self._get_file_line_count(entry)
                size = entry.stat().st_size
                size_str = self._format_file_size(size)
                lines.append(f"  [FILE] {entry.name} ({line_count} lines, {size_str})")

        return "\n".join(lines)

    async def _list_recursive(
        self, root: Path, max_depth: int, include_hidden: bool, current_depth: int = 0
    ) -> str:
        """List directory contents recursively with tree structure.

        Args:
            root: Root directory to list.
            max_depth: Maximum depth for recursive listing.
            include_hidden: Whether to include hidden files.
            current_depth: Current recursion depth.

        Returns:
            Formatted directory tree.
        """

        if current_depth >= max_depth:
            return ""

        entries = list(root.iterdir())

        # Filter hidden files if needed
        if not include_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        # Sort: directories first, then files alphabetically (case-insensitive)
        entries.sort(key=lambda e: (not e.is_dir(), e.name.casefold()))

        # Build output
        lines: list[str] = []
        indent = "  " * current_depth

        for entry in entries:
            if entry.is_dir():
                lines.append(f"{indent}  {entry.name}/")
                # Recurse into subdirectory
                if current_depth + 1 < max_depth:
                    sub_result = await self._list_recursive(
                        entry, max_depth, include_hidden, current_depth + 1
                    )
                    if sub_result:
                        lines.append(sub_result)
            else:
                # Get file metadata
                line_count = self._get_file_line_count(entry)
                lines.append(f"{indent}  {entry.name} ({line_count} lines)")

        return "\n".join(lines)

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _get_file_line_count(self, path: Path) -> int:
        """Get the number of lines in a text file using streaming.

        This method is memory-efficient as it reads the file line-by-line
        instead of loading the entire content into memory.

        Args:
            path: Path to the file.

        Returns:
            Number of lines in the file.
        """
        try:
            with path.open(mode="r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def _format_file_size(self, size: int) -> str:
        """Format file size in human-readable format.

        Args:
            size: File size in bytes.

        Returns:
            Formatted size string (e.g., "5.2 KB").
        """
        if size < self._BYTE_SIZE_THRESHOLD:
            return f"{size} B"

        size_f = float(size)
        for unit in ["KB", "MB", "GB"]:
            size_f /= self._BYTE_SIZE_THRESHOLD
            if size_f < self._BYTE_SIZE_THRESHOLD:
                return f"{size_f:.1f} {unit}"

        return f"{size_f / self._BYTE_SIZE_THRESHOLD:.1f} TB"

    def _truncate_output(self, output: str, is_find_files: bool) -> str:
        """Truncate output to OUTPUT_LIMIT characters.

        Args:
            output: The output string to truncate.
            is_find_files: Whether this is find_files output (for footer message).

        Returns:
            Truncated output with footer message.
        """
        if len(output) <= OUTPUT_LIMIT:
            return output

        truncated = output[:OUTPUT_LIMIT]
        if is_find_files:
            truncated += "\n\n... [file list truncated after 80_000 characters. Use maxResults to limit output]"
        else:
            truncated += "\n\n... [directory listing truncated after 80_000 characters. Use maxDepth to limit recursive depth]"
        return truncated

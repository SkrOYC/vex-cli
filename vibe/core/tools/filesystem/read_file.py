"""ReadFileTool implementation for viewing files with smart outline mode and media support.

This module provides the ReadFileTool class that enables viewing files with intelligent
content/outline mode switching, media file detection, and base64 encoding.

Features:
- Content view mode with line numbers and view_range support
- Outline view mode using ast-grep-py for Python files
- Media file detection and base64 encoding (images and audio)
- Smart view type switching based on file size
- Directory viewing with list and outline modes
- Output truncation at 80,000 characters

Example:
    ```python
    from vibe.core.tools.filesystem.read_file import ReadFileTool, ReadFileArgs

    tool = ReadFileTool(config=tool_config, state=tool_state)
    result = await tool.run(ReadFileArgs(path="test.py", view_type="content"))
    ```
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from vibe.core.tools.base import BaseTool, BaseToolConfig, BaseToolState
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import (
    LARGE_FILE_THRESHOLD,
    MAX_INLINE_MEDIA_BYTES,
    OUTPUT_LIMIT,
    FileSystemError,
)

# Constants for ReadFileTool
UNSUPPORTED_LANGUAGE_PREVIEW_LINES = 100  # Lines to show for unsupported languages
MAX_NAME_PREVIEW_LENGTH = 50  # Max characters for symbol name preview
BYTE_SIZE_THRESHOLD = 1024  # Threshold for size formatting (bytes per KB)

# =============================================================================
# Argument and Result Models
# =============================================================================


class ReadFileArgs(BaseModel):
    """Arguments for the read_file tool.

    Attributes:
        path: File path (absolute or relative to working directory).
        view_type: How to view the file content - "content" shows file with line
            numbers, "outline" shows code structure, "auto" uses smart switching.
        view_range: Optional tuple of [start, end] lines (1-based, -1 for end).
            Only valid for content view mode.
        recursive: Whether to recursively include subdirectories (for directories).
        include_patterns: File patterns to include (for directories with outline mode).
    """

    path: str
    view_type: Literal["content", "outline", "auto"] = "auto"
    view_range: tuple[int, int] | None = None
    recursive: bool = False
    include_patterns: list[str] | None = None


class ReadFileContentResult(BaseModel):
    """Result of reading a text file in content view mode.

    Attributes:
        output: File content with line numbers.
        line_count: Total number of lines in the file.
    """

    output: str
    line_count: int


class ReadFileMediaResult(BaseModel):
    """Result of viewing a media file (image or audio).

    Attributes:
        type: Type of media content ("image" or "audio").
        data: Base64-encoded media data.
        mime_type: MIME type of the media (e.g., "image/png", "audio/mp3").
        path: Absolute path to the file.
        size: File size in bytes.
    """

    type: Literal["image", "audio"]
    data: str
    mime_type: str
    path: str
    size: int


class ReadFileResult(BaseModel):
    """Result of the read_file tool operation.

    This model can hold either content result or media result.

    Attributes:
        content: Content result for text files (optional).
        media: Media result for image/audio files (optional).
        raw: Raw string output for simple cases (optional).
    """

    content: ReadFileContentResult | None = None
    media: ReadFileMediaResult | None = None
    raw: str | None = None

    def __str__(self) -> str:
        """Return string representation for compatibility."""
        if self.content:
            return self.content.output
        if self.media:
            return f"[{self.media.type} file: {self.media.path}]"
        if self.raw:
            return self.raw
        return ""


# =============================================================================
# Tool Configuration and State
# =============================================================================


class ReadFileToolConfig(BaseToolConfig):
    """Configuration for ReadFileTool.

    Extends BaseToolConfig to include the ViewTrackerService dependency.

    Attributes:
        view_tracker: Service for tracking file views during the session.
    """

    view_tracker: ViewTrackerService | None = None

    class Config:
        """Pydantic configuration for arbitrary types."""

        arbitrary_types_allowed = True


class ReadFileToolState(BaseToolState):
    """State for ReadFileTool.

    Currently empty but reserved for future features like caching.

    Attributes:
        (none currently - reserved for future use)
    """

    model_config = ConfigDict(extra="forbid")

    # Reserved for future features like outline caching
    pass


# =============================================================================
# ReadFileTool Implementation
# =============================================================================


class ReadFileTool(
    BaseTool[ReadFileArgs, ReadFileResult, ReadFileToolConfig, ReadFileToolState]
):
    """Tool for viewing files with smart outline mode and media support.

    This tool provides comprehensive file viewing capabilities:
    - Content view: File with line numbers, supports view_range
    - Outline view: Code structure using ast-grep-py (Python only)
    - Media view: Base64-encoded images and audio files
    - Smart switching: Auto mode selects appropriate view based on file size
    - Directory viewing: List or outline mode for directories

    Attributes:
        name: Tool name identifier.
        description: Description of tool functionality.
        args_schema: Pydantic model for input validation.
        config: Tool configuration including ViewTrackerService.
        state: Tool state for tracking operations.
    """

    name = "read_file"
    description = "View files with smart outline mode and media support. Shows file content with line numbers, code structure outlines, or embeds media files."
    args_schema: type[ReadFileArgs] = ReadFileArgs

    # Supported file extensions
    _PYTHON_EXTENSIONS: ClassVar[set[str]] = {".py", ".pyw", ".py3", ".pyi"}
    _IMAGE_EXTENSIONS: ClassVar[set[str]] = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".bmp",
    }
    _AUDIO_EXTENSIONS: ClassVar[set[str]] = {
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".aac",
        ".m4a",
    }

    async def run(self, args: ReadFileArgs) -> ReadFileResult:
        """Execute the read_file tool operation.

        Determines whether to view a file or directory, and selects the appropriate
        view mode based on the arguments and file type.

        Args:
            args: ReadFileArgs containing path and view options.

        Returns:
            ReadFileResult with content, outline, or media data.

        Raises:
            FileSystemError: If path resolution or file operations fail.
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(args.path)

        # Check if path exists
        if not resolved_path.exists():
            raise FileSystemError(
                message=f"File not found: '{args.path}'",
                code="FILE_NOT_FOUND",
                path=str(resolved_path),
                suggestions=[
                    f"The specified path doesn't exist in current working directory: {self.config.effective_workdir}",
                    "Use 'list' command to explore available directories",
                    "Check if the path is correct",
                ],
            )

        # Check if path is a directory
        if resolved_path.is_dir():
            result = await self._view_directory(resolved_path, args)
            return ReadFileResult(content=result)

        # Path is a file - check if it's a media file
        if self._is_media_file(resolved_path):
            result = await self._view_media_file(resolved_path)
            return ReadFileResult(media=result)

        # It's a text file - determine view type and view
        effective_view_type = self._determine_effective_view_type(
            resolved_path, args.view_type, args.view_range
        )
        result = await self._view_text_file(
            resolved_path, effective_view_type, args.view_range
        )
        return ReadFileResult(content=result)

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
    # View Type Determination
    # =============================================================================

    def _determine_effective_view_type(
        self, path: Path, view_type: str, view_range: tuple[int, int] | None
    ) -> Literal["content", "outline"]:
        """Determine the effective view type based on arguments and file size.

        Args:
            path: Path to the file.
            view_type: User-specified view type.
            view_range: Optional view range.

        Returns:
            The effective view type to use.
        """
        # If explicitly set to content or outline, use that
        if view_type in {"content", "outline"}:
            return view_type  # type: ignore[return-value]

        # Auto mode: view_range takes precedence
        if view_range is not None:
            return "content"

        # Auto mode: check file size
        try:
            file_size = path.stat().st_size
            if file_size > LARGE_FILE_THRESHOLD:
                return "outline"
            return "content"
        except OSError:
            # If we can't get file size, default to content
            return "content"

    # =============================================================================
    # File Type Detection
    # =============================================================================

    def _is_python_file(self, path: Path) -> bool:
        """Check if the file has a Python extension.

        Args:
            path: Path to the file.

        Returns:
            True if the file has a Python extension.
        """
        return path.suffix.lower() in self._PYTHON_EXTENSIONS

    def _is_media_file(self, path: Path) -> bool:
        """Check if the file is a media file (image or audio).

        Args:
            path: Path to the file.

        Returns:
            True if the file is a media file.
        """
        suffix = path.suffix.lower()
        return suffix in self._IMAGE_EXTENSIONS or suffix in self._AUDIO_EXTENSIONS

    def _is_image_file(self, path: Path) -> bool:
        """Check if the file is an image file.

        Args:
            path: Path to the file.

        Returns:
            True if the file is an image file.
        """
        return path.suffix.lower() in self._IMAGE_EXTENSIONS

    def _is_audio_file(self, path: Path) -> bool:
        """Check if the file is an audio file.

        Args:
            path: Path to the file.

        Returns:
            True if the file is an audio file.
        """
        return path.suffix.lower() in self._AUDIO_EXTENSIONS

    def _get_mime_type(self, path: Path) -> str:
        """Get the MIME type for a file.

        Args:
            path: Path to the file.

        Returns:
            MIME type string (e.g., "image/png", "audio/mp3").
        """
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            return mime_type

        # Fallback for common types
        suffix = path.suffix.lower()
        if suffix in self._IMAGE_EXTENSIONS:
            image_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }
            return image_types.get(suffix, "image/png")

        if suffix in self._AUDIO_EXTENSIONS:
            audio_types = {
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
                ".flac": "audio/flac",
                ".aac": "audio/aac",
                ".m4a": "audio/mp4",
            }
            return audio_types.get(suffix, "audio/mpeg")

        return "application/octet-stream"

    def _get_file_line_count(self, path: Path) -> int:
        """Get the number of lines in a text file.

        Args:
            path: Path to the file.

        Returns:
            Number of lines in the file.
        """
        try:
            content = path.read_text(encoding="utf-8")
            return len(content.split("\n"))
        except (OSError, UnicodeDecodeError):
            return 0

    # =============================================================================
    # Content View Mode
    # =============================================================================

    async def _view_text_file(
        self,
        path: Path,
        view_type: Literal["content", "outline"],
        view_range: tuple[int, int] | None,
    ) -> ReadFileContentResult:
        """View a text file in content or outline mode.

        Args:
            path: Path to the file.
            view_type: View mode to use.
            view_range: Optional view range [start, end] (1-based, -1 for end).

        Returns:
            ReadFileContentResult with formatted content.

        Raises:
            FileSystemError: If view_range is invalid.
        """
        # Read file content
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            raise FileSystemError(
                message=f"Permission denied: '{path}'",
                code="PERMISSION_DENIED",
                path=str(path),
                suggestions=[
                    "This file requires elevated permissions",
                    "Try checking file permissions or running with appropriate access rights",
                ],
            ) from e

        lines = content.split("\n")
        line_count = len(lines)

        # Validate view_range if provided
        if view_range is not None:
            start, end = view_range
            # Convert 1-based to 0-based indices
            start_idx = max(0, start - 1)
            # Handle -1 for end
            end_idx = line_count if end == -1 else end

            # Validate bounds
            if start_idx >= line_count or end_idx > line_count:
                max_start = max(1, line_count - 49)
                raise FileSystemError(
                    message=f"View range [{start}, {end}] exceeds file length ({line_count} lines)",
                    code="VIEW_RANGE_OUT_OF_BOUNDS",
                    path=str(path),
                    suggestions=[
                        f"Valid range: [1, {line_count}] for entire file",
                        f"Use -1 for end parameter to view from line {start} to end of file",
                        f"Example: [{max_start}, {line_count}] for last 50 lines",
                        "Tip: Use 'list' command first to understand file sizes before viewing",
                    ],
                )

            if start_idx > end_idx:
                raise FileSystemError(
                    message=f"Invalid view range: start ({start}) > end ({end})",
                    code="INVALID_VIEW_RANGE",
                    path=str(path),
                    suggestions=["Use start <= end for view range"],
                )

            # Extract the requested range
            lines = lines[start_idx:end_idx]

        if view_type == "outline":
            # Generate outline for Python files
            if self._is_python_file(path):
                outline = self._generate_python_outline(content, str(path))
                if outline:
                    output = self._format_outline(outline, str(path))
                else:
                    # Fallback to content view if outline generation fails
                    output = self._format_content_with_line_numbers(
                        lines, start_line=view_range[0] if view_range else 1
                    )
                    output += "\n\n[Note: Could not generate outline, showing content]"
            else:
                # Non-Python files: show first 100 lines as outline fallback
                output = self._format_content_with_line_numbers(
                    lines[:UNSUPPORTED_LANGUAGE_PREVIEW_LINES],
                    start_line=view_range[0] if view_range else 1,
                )
                if len(lines) > UNSUPPORTED_LANGUAGE_PREVIEW_LINES:
                    output += f"\n\n[Unsupported language, showing first {UNSUPPORTED_LANGUAGE_PREVIEW_LINES} lines]"
        else:
            # Content view mode
            output = self._format_content_with_line_numbers(
                lines, start_line=view_range[0] if view_range else 1
            )

        # Apply output truncation
        output = self._truncate_output(output)

        # Record view
        if self.config.view_tracker is not None:
            self.config.view_tracker.record_view(str(path))

        return ReadFileContentResult(output=output, line_count=line_count)

    def _format_content_with_line_numbers(
        self, lines: list[str], start_line: int = 1
    ) -> str:
        """Format content with line numbers.

        Args:
            lines: List of lines to format.
            start_line: The starting line number (1-based). Defaults to 1.

        Returns:
            Formatted string with line numbers.
        """
        # Calculate padding based on total expected lines
        total_lines = start_line + len(lines) - 1
        padding = len(str(total_lines))

        formatted_lines = []
        for i, line in enumerate(lines, start=start_line):
            line_num = str(i).rjust(padding)
            formatted_lines.append(f"{line_num}→{line}")

        return "\n".join(formatted_lines)

    def _truncate_output(self, output: str) -> str:
        """Truncate output to OUTPUT_LIMIT characters.

        Args:
            output: The output string to truncate.

        Returns:
            Truncated output with footer message.
        """
        if len(output) <= OUTPUT_LIMIT:
            return output

        truncated = output[:OUTPUT_LIMIT]
        truncated += "\n\n... [file content truncated after 80_000 characters. Use viewRange to limit output or viewType='outline' for summary]"
        return truncated

    # =============================================================================
    # Outline Generation (Python only using ast-grep-py)
    # =============================================================================

    def _generate_python_outline(self, content: str, file_path: str) -> list[dict]:
        """Generate a code structure outline for a Python file using ast-grep-py.

        Args:
            content: Python source code.
            file_path: Path to the file (for reference).

        Returns:
            List of outline entries with name, kind, and line range.
        """
        try:
            from ast_grep_py import SgRoot

            root = SgRoot(content, "python")
            node = root.root()

            outline: list[dict] = []

            # Add all symbol types
            outline.extend(
                self._extract_symbols_by_kind(node, "class_definition", "class")
            )
            outline.extend(
                self._extract_symbols_by_kind(node, "function_definition", "function")
            )
            outline.extend(self._extract_async_functions(node))
            outline.extend(
                self._extract_symbols_by_kind(node, "import_statement", "import")
            )
            outline.extend(
                self._extract_symbols_by_kind(node, "import_from_statement", "import")
            )
            outline.extend(
                self._extract_symbols_by_kind(node, "expression_statement", "variable")
            )

            return outline

        except ImportError:
            # ast-grep-py not available, return empty outline
            return []
        except Exception:
            # On any error, return empty outline
            return []

    def _extract_symbols_by_kind(
        self, node: Any, kind: str, symbol_kind: str
    ) -> list[dict]:
        """Extract symbols of a specific kind from the AST.

        Args:
            node: The root AST node.
            kind: The ast-grep kind to search for.
            symbol_kind: The symbol type name for the outline.

        Returns:
            List of symbol information dictionaries.
        """
        results: list[dict] = []
        for ast_node in node.find_all(kind=kind):
            symbol_info = self._extract_symbol_info(ast_node, symbol_kind)
            if symbol_info:
                results.append(symbol_info)
        return results

    def _extract_async_functions(self, node: Any) -> list[dict]:
        """Extract async functions from the AST.

        Args:
            node: The root AST node.

        Returns:
            List of async function symbol information dictionaries.
        """
        results: list[dict] = []
        for func_node in node.find_all(kind="function_definition"):
            try:
                if func_node.text().strip().startswith("async "):
                    symbol_info = self._extract_symbol_info(func_node, "async function")
                    if symbol_info:
                        results.append(symbol_info)
            except Exception:
                pass
        return results

    def _extract_symbol_info(self, node: Any, kind: str) -> dict[str, int | str] | None:
        """Extract symbol information from an AST node.

        Args:
            node: The ast-grep SgNode.
            kind: The kind of symbol (class, function, import, etc.).

        Returns:
            Dictionary with name, kind, startLine, and endLine, or None if extraction fails.
        """
        try:
            # Get the range information
            rng = node.range()
            start_line = rng.start.line + 1  # Convert to 1-based
            end_line = rng.end.line + 1  # Convert to 1-based

            # Try to get the name
            name = None

            # Try to get name field
            try:
                name_node = node.field("name")
                if name_node:
                    name = name_node.text()
            except Exception:
                pass

            # If no name field, try to get the text directly
            if not name:
                try:
                    text = node.text()
                    # Take first line and truncate if too long
                    first_line = text.split("\n")[0]
                    name = (
                        first_line[:MAX_NAME_PREVIEW_LENGTH] + "..."
                        if len(first_line) > MAX_NAME_PREVIEW_LENGTH
                        else first_line
                    )
                except Exception:
                    name = kind

            if not name:
                return None

            return {
                "name": name,
                "kind": kind,
                "startLine": start_line,
                "endLine": end_line,
            }

        except Exception:
            return None

    def _format_outline(self, outline: list[dict], file_path: str) -> str:
        """Format outline entries into a readable string.

        Args:
            outline: List of outline entries.
            file_path: Path to the file (for header).

        Returns:
            Formatted outline string.
        """
        if not outline:
            return f"FILE: {file_path}\n\n[No structural elements found]"

        # Sort by line number
        sorted_outline = sorted(outline, key=lambda x: x.get("startLine", 999))

        # Format each entry
        lines = [f"FILE: {file_path}", ""]

        for entry in sorted_outline:
            kind = entry.get("kind", "unknown")
            name = entry.get("name", "unknown")
            start = entry.get("startLine", 0)
            end = entry.get("endLine", 0)

            kind_prefix = {
                "class": "[CLASS]",
                "function": "[FUNC]",
                "async function": "[ASYNC]",
                "import": "[IMPORT]",
                "variable": "[VAR]",
            }.get(kind, f"[{kind.upper()}]")

            lines.append(f"  {kind_prefix} {name}:{start}-{end}")

        return "\n".join(lines)

    # =============================================================================
    # Media File Handling
    # =============================================================================

    async def _view_media_file(self, path: Path) -> ReadFileMediaResult:
        """View a media file (image or audio) with base64 encoding.

        Args:
            path: Path to the media file.

        Returns:
            ReadFileMediaResult with base64-encoded data.

        Raises:
            FileSystemError: If file is too large to embed.
        """
        # Get file size
        try:
            file_size = path.stat().st_size
        except OSError as e:
            raise FileSystemError(
                message=f"Permission denied: '{path}'",
                code="PERMISSION_DENIED",
                path=str(path),
                suggestions=[
                    "This file requires elevated permissions",
                    "Try checking file permissions or running with appropriate access rights",
                ],
            ) from e

        # Check size limit
        if file_size > MAX_INLINE_MEDIA_BYTES:
            media_type = "image" if self._is_image_file(path) else "audio"
            size_str = self._format_file_size(file_size)
            limit_str = self._format_file_size(MAX_INLINE_MEDIA_BYTES)

            # Record view even for large files
            if self.config.view_tracker is not None:
                self.config.view_tracker.record_view(str(path))

            # Return guidance for large files
            raise FileSystemError(
                message=f"Inline {media_type} preview skipped: {path.name} is {size_str}, exceeding {limit_str} inline limit",
                code="MEDIA_TOO_LARGE",
                path=str(path),
                suggestions=[
                    "Download file directly, or reduce its size before viewing inline",
                    f"Absolute path: {path}",
                ],
            )

        # Read and encode the file
        try:
            file_bytes = path.read_bytes()
            encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        except OSError as e:
            raise FileSystemError(
                message=f"Permission denied: '{path}'",
                code="PERMISSION_DENIED",
                path=str(path),
                suggestions=[
                    "This file requires elevated permissions",
                    "Try checking file permissions or running with appropriate access rights",
                ],
            ) from e

        # Determine media type
        media_type: Literal["image", "audio"] = (
            "image" if self._is_image_file(path) else "audio"
        )

        # Get MIME type
        mime_type = self._get_mime_type(path)

        # Record view
        if self.config.view_tracker is not None:
            self.config.view_tracker.record_view(str(path))

        return ReadFileMediaResult(
            type=media_type,
            data=encoded_data,
            mime_type=mime_type,
            path=str(path),
            size=file_size,
        )

    def _format_file_size(self, size: int) -> str:
        """Format file size in human-readable format.

        Args:
            size: File size in bytes.

        Returns:
            Formatted size string (e.g., "5.2 MB").
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if size < BYTE_SIZE_THRESHOLD:
                return f"{size:.1f} {unit}"
            size = int(size // BYTE_SIZE_THRESHOLD)
        return f"{size:.1f} TB"

    # =============================================================================
    # Directory Viewing
    # =============================================================================

    async def _view_directory(
        self, path: Path, args: ReadFileArgs
    ) -> ReadFileContentResult:
        """View a directory in list or outline mode.

        Args:
            path: Path to the directory.
            args: ReadFileArgs with view options.

        Returns:
            ReadFileContentResult with directory listing.

        Raises:
            FileSystemError: If view_range is specified for a directory.
        """
        # view_range cannot be used with directories
        if args.view_range is not None:
            raise FileSystemError(
                message="viewRange cannot be used with directories",
                code="INVALID_ARGUMENT",
                path=str(path),
                suggestions=[
                    'Use viewType="outline" to get code structure overview',
                    'Use "list" command for directory contents and file metadata',
                    "Remove viewRange parameter when viewing directories",
                ],
            )

        try:
            entries = list(path.iterdir())
        except OSError as e:
            raise FileSystemError(
                message=f"Permission denied: '{path}'",
                code="PERMISSION_DENIED",
                path=str(path),
                suggestions=[
                    "This directory requires elevated permissions",
                    "Try checking directory permissions or running with appropriate access rights",
                ],
            ) from e

        if args.view_type == "outline":
            return await self._view_directory_outline(path, entries, args)
        else:
            return self._view_directory_list(path, entries)

    def _view_directory_list(
        self, path: Path, entries: list[Path]
    ) -> ReadFileContentResult:
        """View directory contents as a simple list.

        Args:
            path: Path to the directory.
            entries: List of entries in the directory.

        Returns:
            ReadFileContentResult with directory listing.
        """
        lines = [f"DIRECTORY: {path}", ""]

        # Sort entries: directories first, then files, alphabetically
        sorted_entries = sorted(entries, key=lambda e: (not e.is_dir(), e.name))

        for entry in sorted_entries:
            if entry.is_dir():
                lines.append(f"  [DIR]  {entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                    size_str = self._format_file_size(size)
                    lines.append(f"  [FILE] {entry.name} ({size_str})")
                except OSError:
                    lines.append(f"  [FILE] {entry.name}")

        output = "\n".join(lines)
        line_count = len(entries)

        # Apply output truncation
        if len(output) > OUTPUT_LIMIT:
            output = output[:OUTPUT_LIMIT]
            output += "\n\n... [directory listing truncated after 80_000 characters]"

        # Record view
        if self.config.view_tracker is not None:
            self.config.view_tracker.record_view(str(path))

        return ReadFileContentResult(output=output, line_count=line_count)

    async def _view_directory_outline(
        self, path: Path, entries: list[Path], args: ReadFileArgs
    ) -> ReadFileContentResult:
        """View directory contents with code outlines for Python files.

        Args:
            path: Path to the directory.
            entries: List of entries in the directory.
            args: ReadFileArgs with view options.

        Returns:
            ReadFileContentResult with directory outline.
        """
        lines = [f"DIRECTORY OUTLINE: {path}", ""]

        # Filter for Python files only in outline mode
        python_files = [e for e in entries if e.is_file() and self._is_python_file(e)]

        if not python_files:
            lines.append("  [No Python files found in directory]")
            output = "\n".join(lines)

            if self.config.view_tracker is not None:
                self.config.view_tracker.record_view(str(path))

            return ReadFileContentResult(output=output, line_count=len(entries))

        # Process Python files in batches
        DIRECTORY_OUTLINE_BATCH_SIZE = 50
        batch_size = DIRECTORY_OUTLINE_BATCH_SIZE

        for i in range(0, len(python_files), batch_size):
            batch = python_files[i : i + batch_size]

            for py_file in batch:
                try:
                    content = py_file.read_text(encoding="utf-8")
                    file_outline = self._generate_python_outline(content, str(py_file))

                    if file_outlines := self._format_file_outline(
                        py_file, file_outline
                    ):
                        lines.append("")
                        lines.append(f"FILE: {py_file.name}")
                        lines.append(file_outlines)
                except (OSError, UnicodeDecodeError):
                    # Skip files that can't be read
                    continue

        output = "\n".join(lines)
        line_count = len(python_files)

        # Apply output truncation
        if len(output) > OUTPUT_LIMIT:
            output = output[:OUTPUT_LIMIT]
            output += "\n\n... [directory outline truncated after 80_000 characters]"

        # Record view
        if self.config.view_tracker is not None:
            self.config.view_tracker.record_view(str(path))

        return ReadFileContentResult(output=output, line_count=line_count)

    def _format_file_outline(self, path: Path, outline: list[dict]) -> str | None:
        """Format outline for a single file.

        Args:
            path: Path to the file.
            outline: List of outline entries.

        Returns:
            Formatted outline string, or None if no entries.
        """
        if not outline:
            return None

        # Sort by line number
        sorted_outline = sorted(outline, key=lambda x: x.get("startLine", 999))

        lines = []
        for entry in sorted_outline:
            kind = entry.get("kind", "unknown")
            name = entry.get("name", "unknown")
            start = entry.get("startLine", 0)
            end = entry.get("endLine", 0)

            kind_prefix = {
                "class": "[CLASS]",
                "function": "[FUNC]",
                "async function": "[ASYNC]",
                "import": "[IMPORT]",
                "variable": "[VAR]",
            }.get(kind, f"[{kind.upper()}]")

            lines.append(f"    {kind_prefix} {name}:{start}-{end}")

        return "\n".join(lines)

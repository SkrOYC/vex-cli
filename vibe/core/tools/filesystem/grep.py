"""GrepTool implementation for searching files with regex support and context lines.

This module provides the GrepTool class that enables file content search with:
- Regex and literal search modes
- Case-sensitive and case-insensitive matching
- File pattern filtering (glob patterns)
- Recursive and non-recursive search
- Context lines before and after matches
- Output formatting matching TypeScript FileExplorer

Example:
    ```python
    from vibe.core.tools.filesystem.grep import GrepTool, GrepArgs

    tool = GrepTool(config=tool_config, state=tool_state)
    result = await tool.run(GrepArgs(
        path="src",
        query="function.*Component",
        regex=True,
        case_sensitive=False
    ))
    ```
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import ClassVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.base import BaseTool, BaseToolConfig, BaseToolState
from vibe.core.tools.filesystem.types import (
    DEFAULT_CONTEXT_AFTER,
    DEFAULT_CONTEXT_BEFORE,
    OUTPUT_LIMIT,
)

# =============================================================================
# Constants
# =============================================================================

# ASCII printable character range
ASCII_PRINTABLE_MIN: int = 32
ASCII_PRINTABLE_MAX: int = 127
# Minimum ratio of ASCII printable characters for text file detection
TEXT_ASCII_RATIO_THRESHOLD: float = 0.7
# Minimum header bytes to read for text detection
TEXT_DETECTION_HEADER_BYTES: int = 1024

# =============================================================================
# Argument and Result Models
# =============================================================================


class GrepArgs(BaseModel):
    """Arguments for the grep tool.

    Attributes:
        path: Directory path to search (absolute or relative to working directory).
        query: Search query or regex pattern to search for.
        patterns: List of file patterns to limit search (e.g., ['*.py', '*.ts']).
        case_sensitive: Whether to perform case-sensitive search (default: False).
        regex: Whether to interpret query as regex pattern (default: False).
        recursive: Whether to search subdirectories (default: True).
        max_results: Maximum number of matches to return (default: 1000, range: 1-10000).
        include_hidden: Whether to include hidden files and directories (default: False).
    """

    path: str = "."
    query: str
    patterns: list[str] = Field(default_factory=list)
    case_sensitive: bool = False
    regex: bool = False
    recursive: bool = True
    max_results: int = Field(default=1000, ge=1, le=10_000)
    include_hidden: bool = False


class GrepResult(BaseModel):
    """Result of the grep tool operation.

    Attributes:
        output: Formatted search results with context lines.
    """

    output: str


# =============================================================================
# Tool Configuration and State
# =============================================================================


class GrepToolConfig(BaseToolConfig):
    """Configuration for GrepTool.

    Extends BaseToolConfig without additional fields.
    """

    model_config = ConfigDict(extra="allow")


class GrepToolState(BaseToolState):
    """State for GrepTool.

    Currently empty - grep is a read-only search tool with no persistent state.

    Attributes:
        (none currently - read-only tool)
    """

    model_config = ConfigDict(extra="forbid")

    pass


# =============================================================================
# TypedDicts for type-safe search results
# =============================================================================


class SearchMatchContext(TypedDict, total=False):
    """Context lines around a match."""

    before: list[str]
    after: list[str]


class SearchMatch(TypedDict):
    """A single search match with context."""

    line: int
    column: int
    text: str
    context: SearchMatchContext | None


class SearchResult(TypedDict):
    """Search result for a single file."""

    path: str
    matches: list[SearchMatch]


# =============================================================================
# GrepTool Implementation
# =============================================================================


class GrepTool(BaseTool[GrepArgs, GrepResult, GrepToolConfig, GrepToolState]):
    """Tool for searching file content with regex support and context lines.

    This tool provides comprehensive search capabilities:
    - Regex and literal search modes
    - Case-sensitive and case-insensitive matching
    - File pattern filtering using glob patterns
    - Recursive and non-recursive directory search
    - Context lines before and after matches (5 before, 3 after by default)
    - Output formatting matching TypeScript FileExplorer behavior

    Attributes:
        name: Tool name identifier ("grep").
        description: Description of tool functionality.
        args_schema: Pydantic model for input validation.
        config: Tool configuration.
        state: Tool state for tracking operations.
    """

    name = "grep"
    description = "Search files with regex support and context lines. Use regex=true for regex patterns, or literal search by default."
    args_schema: type[GrepArgs] = GrepArgs

    # Text file extensions for filtering
    _TEXT_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rs",
        ".go",
        ".kt",
        ".swift",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".html",
        ".css",
        ".xml",
        ".sql",
        ".rb",
        ".pl",
        ".php",
        ".lua",
        ".dart",
        ".elm",
        ".ex",
        ".exs",
        ".ml",
        ".mli",
        ".hs",
        ".lhs",
        ".clj",
        ".cljs",
        ".lisp",
        ".scm",
        ".r",
        ".jl",
        ".m",
        ".nb",
    })

    async def run(self, args: GrepArgs) -> GrepResult:
        """Execute the grep tool operation.

        Args:
            args: GrepArgs containing search path, query, and options.

        Returns:
            GrepResult with formatted search output.

        Raises:
            FileSystemError: If path resolution or file operations fail.
            ValueError: If regex pattern is invalid.
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(args.path)

        # Validate path exists
        if not resolved_path.exists():
            raise ValueError(
                f"Directory not found: '{args.path}'\n\n"
                f"The specified directory doesn't exist in current working directory: {self.config.effective_workdir}\n"
                "Try using 'list' command to explore available directories."
            )

        # Check if path is a directory
        if not resolved_path.is_dir():
            raise ValueError(
                f"Path is not a directory: '{args.path}'\n\n"
                "Use 'view' command to examine file contents.\n"
                "Specify a directory path to search."
            )

        try:
            # Get text files to search
            text_files = await self._get_text_files_for_search(resolved_path, args)

            if not text_files:
                raise ValueError(
                    f"No text files found in: '{args.path}'\n\n"
                    "The search only works on text files. Try:\n"
                    "• Use 'list' command to see what files are in directory\n"
                    "• Specify a directory with known text files\n"
                    "• Add file patterns to search specific file types"
                )

            # Execute search
            results = await self._execute_search(text_files, args)

            if not results:
                return GrepResult(
                    output=f'No matches found for "{args.query}" in {len(text_files)} files\n\n'
                    f"Try adjusting your search pattern or check the file paths."
                )

            # Format output
            output = self._format_search_results(results)

            # Add truncation footer if needed
            total_matches = sum(len(r["matches"]) for r in results)
            if total_matches >= args.max_results:
                output += f"\n\nFound {total_matches} matches across files (showing first {args.max_results} matches). Consider using more specific search patterns or file filters to narrow results."

            # Apply output truncation
            output = self._truncate_output(output)

            return GrepResult(output=output)

        except ValueError as e:
            # Re-raise regex errors with helpful message
            error_msg = str(e)
            if "Invalid regex pattern" in error_msg:
                # Wrap the regex error with helpful message
                raise ValueError(
                    f"{error_msg}\n\n"
                    "The provided regex pattern is invalid. Check for:\n"
                    "• Unmatched parentheses or brackets\n"
                    "• Incomplete escape sequences\n"
                    "• Invalid quantifiers\n\n"
                    "Try escaping special characters with regex=false for literal search."
                ) from e
            # For other errors (like "No text files found"), re-raise as-is
            raise

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
    # Text File Detection
    # =============================================================================

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is likely a text file using heuristics.

        Args:
            file_path: Path to check.

        Returns:
            True if file is likely a text file, False otherwise.
        """
        # Check file extension first
        if file_path.suffix.lower() in self._TEXT_EXTENSIONS:
            return True

        # Check by reading first few bytes
        try:
            with file_path.open("rb") as f:
                header = f.read(TEXT_DETECTION_HEADER_BYTES)

                # Check for null bytes (binary indicator)
                if b"\x00" in header:
                    return False

                # Check for high ratio of non-ASCII characters
                text_chars = sum(
                    1 for b in header if ASCII_PRINTABLE_MIN <= b < ASCII_PRINTABLE_MAX
                )
                if len(header) > 0:
                    return text_chars / len(header) > TEXT_ASCII_RATIO_THRESHOLD

                return True
        except OSError:
            return False

    # =============================================================================
    # File Discovery
    # =============================================================================

    def _discover_files_by_pattern(
        self, search_path: Path, patterns: list[str], recursive: bool
    ) -> set[Path]:
        """Discover files matching the given patterns.

        Args:
            search_path: Directory to search in.
            patterns: List of file patterns to match.
            recursive: Whether to search recursively.

        Returns:
            Set of file paths matching the patterns.
        """
        candidate_files: set[Path] = set()
        glob_method = search_path.rglob if recursive else search_path.glob

        for pattern in patterns:
            for file_path in glob_method(pattern):
                if file_path.is_file():
                    candidate_files.add(file_path)

        return candidate_files

    def _filter_hidden_files(
        self, candidate_files: set[Path], search_path: Path, include_hidden: bool
    ) -> set[Path]:
        """Filter out hidden files and files in hidden directories.

        Args:
            candidate_files: Set of file paths to filter.
            search_path: The base search directory.
            include_hidden: Whether to include hidden files.

        Returns:
            Filtered set of file paths.
        """
        if include_hidden:
            return candidate_files

        filtered_files: set[Path] = set()
        for file_path in candidate_files:
            try:
                relative_path = file_path.relative_to(search_path)
                # Check if any path component starts with dot
                if any(part.startswith(".") for part in relative_path.parts):
                    continue
            except ValueError:
                # Fallback for paths not directly under search_path (e.g., symlinks)
                if file_path.name.startswith("."):
                    continue
            filtered_files.add(file_path)

        return filtered_files

    async def _get_text_files_for_search(
        self, search_path: Path, args: GrepArgs
    ) -> list[Path]:
        """Get text files to search, filtered by patterns.

        Args:
            search_path: Directory to search in.
            args: GrepArgs with search options.

        Returns:
            List of text file paths to search.
        """
        # Discover files by pattern
        if args.patterns:
            candidate_files = self._discover_files_by_pattern(
                search_path, args.patterns, args.recursive
            )
        else:
            # No patterns - search all files
            candidate_files = self._discover_files_by_pattern(
                search_path, ["*"], args.recursive
            )

        # Filter hidden files and directories
        candidate_files = self._filter_hidden_files(
            candidate_files, search_path, args.include_hidden
        )

        # Filter text files
        text_files: list[Path] = [
            fp for fp in candidate_files if self._is_text_file(fp)
        ]

        # Sort alphabetically (case-insensitive)
        text_files.sort(key=lambda p: str(p).casefold())

        return text_files

    # =============================================================================
    # Regex Creation
    # =============================================================================

    def _create_search_regex(
        self, query: str, case_sensitive: bool, use_regex: bool
    ) -> re.Pattern[str]:
        """Create search regex from query.

        Args:
            query: Search query string.
            case_sensitive: Whether to use case-sensitive matching.
            use_regex: Whether to treat query as regex pattern.

        Returns:
            Compiled regex pattern.

        Raises:
            ValueError: If regex pattern is invalid.
        """
        flags = re.MULTILINE | re.DOTALL
        if not case_sensitive:
            flags |= re.IGNORECASE

        if use_regex:
            try:
                return re.compile(query, flags)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {query}") from e
        else:
            # Escape special regex characters for literal search
            escaped = re.escape(query)
            return re.compile(escaped, flags)

    # =============================================================================
    # Content Search
    # =============================================================================

    async def _execute_search(
        self, files: list[Path], args: GrepArgs
    ) -> list[SearchResult]:
        """Execute search across multiple files.

        Args:
            files: List of text file paths to search.
            args: GrepArgs with search options.

        Returns:
            List of search results with file path and matches.
        """
        results: list[SearchResult] = []
        total_matches = 0

        regex = self._create_search_regex(args.query, args.case_sensitive, args.regex)

        for file_path in files:
            # Calculate remaining capacity for this file
            remaining_capacity = args.max_results - total_matches
            if remaining_capacity <= 0:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                file_matches = self._search_file_content(
                    content, regex, remaining_capacity
                )

                if file_matches:
                    results.append({"path": str(file_path), "matches": file_matches})
                    total_matches += len(file_matches)
            except OSError:
                # Skip files that can't be read
                continue

        return results

    def _search_file_content(
        self, content: str, regex: re.Pattern[str], max_results: int
    ) -> list[SearchMatch]:
        """Search content for regex matches with context.

        Supports multi-line regex patterns by searching the full content.

        Args:
            content: File content to search.
            regex: Compiled regex pattern.
            max_results: Maximum number of matches to return.

        Returns:
            List of match dictionaries with line, column, text, and context.
        """
        matches: list[SearchMatch] = []
        lines = content.split("\n")

        # Use finditer on full content to support multi-line regex patterns
        for match in regex.finditer(content):
            if len(matches) >= max_results:
                break

            # Calculate line number (1-based)
            match_start = match.start()
            line_number = content.count("\n", 0, match_start) + 1

            # Calculate column number (1-based) - position within the line
            last_newline = content.rfind("\n", 0, match_start)
            if last_newline == -1:
                column_number = match_start + 1
            else:
                column_number = match_start - last_newline

            # Get the line content containing the match
            line_content = (
                lines[line_number - 1] if 0 <= line_number - 1 < len(lines) else ""
            )

            # Build context lines
            context = self._build_context_lines(
                lines, line_number - 1, DEFAULT_CONTEXT_BEFORE, DEFAULT_CONTEXT_AFTER
            )

            matches.append({
                "line": line_number,
                "column": column_number,
                "text": line_content,
                "context": context,
            })

        return matches

    def _build_context_lines(
        self,
        lines: list[str],
        current_index: int,
        context_before: int,
        context_after: int,
    ) -> SearchMatchContext | None:
        """Build context lines before and after a match.

        Args:
            lines: All lines from the file.
            current_index: Index of the matched line.
            context_before: Number of lines before the match.
            context_after: Number of lines after the match.

        Returns:
            Dictionary with 'before' and 'after' lists, or None if no context.
        """
        if context_before <= 0 and context_after <= 0:
            return None

        before: list[str] = []
        after: list[str] = []

        if context_before > 0:
            start = max(0, current_index - context_before)
            before = lines[start:current_index]

        if context_after > 0:
            end = min(len(lines), current_index + 1 + context_after)
            after = lines[current_index + 1 : end]

        return {"before": before, "after": after}

    # =============================================================================
    # Output Formatting
    # =============================================================================

    def _format_search_results(self, results: list[SearchResult]) -> str:
        """Format search results matching TypeScript FileExplorer behavior.

        Args:
            results: List of search results with path and matches.

        Returns:
            Formatted search results string.
        """
        output = ""

        for i, result in enumerate(results):
            is_last = i == len(results) - 1
            output += self._format_single_result_deduped(result, is_last)

        return output

    def _format_single_result_deduped(self, result: SearchResult, is_last: bool) -> str:
        """Format a single search result with deduplicated context lines.

        Args:
            result: Search result with path and matches.
            is_last: Whether this is the last result (no separator after).

        Returns:
            Formatted result string.
        """
        path = result["path"]
        matches = result["matches"]

        # Use first match's line number for file header suffix
        if matches:
            first_match_line = matches[0]["line"]
            output = f"{path}:{first_match_line}\n"
        else:
            output = f"{path}\n"

        if not matches:
            return output

        # Sort matches by line number
        sorted_matches = sorted(matches, key=lambda m: m["line"])

        # Track displayed lines to avoid duplicates
        displayed_lines: set[int] = set()

        # Calculate line number width for alignment
        max_line_num_width = self._calculate_line_number_width(matches)

        for match in sorted_matches:
            output += self._format_match_context(
                match, displayed_lines, max_line_num_width
            )

        if not is_last:
            output += "\n"

        return output

    def _format_match_context(
        self, match: SearchMatch, displayed_lines: set[int], max_line_num_width: int
    ) -> str:
        """Format a match with its context lines.

        Args:
            match: Match dictionary with line, column, text, and context.
            displayed_lines: Set of already displayed line numbers.
            max_line_num_width: Width for line number alignment.

        Returns:
            Formatted match with context.
        """
        output = ""
        context = match.get("context")
        line_num = match["line"]
        padded_num = str(line_num).rjust(max_line_num_width)

        # Format context before (with leading spaces for alignment)
        if context:
            before_lines = context.get("before", [])
            start_line = line_num - len(before_lines)
            for j, ctx_line in enumerate(before_lines):
                line_number = start_line + j
                if line_number not in displayed_lines:
                    output += (
                        f"  {str(line_number).rjust(max_line_num_width)} {ctx_line}\n"
                    )
                    displayed_lines.add(line_number)

        # Format the match line (with > prefix)
        if line_num not in displayed_lines:
            output += f"> {padded_num} {match['text']}\n"
            displayed_lines.add(line_num)

        # Format context after (with leading spaces for alignment)
        if context:
            after_lines = context.get("after", [])
            start_line = line_num + 1
            for j, ctx_line in enumerate(after_lines):
                line_number = start_line + j
                if line_number not in displayed_lines:
                    output += (
                        f"  {str(line_number).rjust(max_line_num_width)} {ctx_line}\n"
                    )
                    displayed_lines.add(line_number)

        return output

    def _calculate_line_number_width(self, matches: list[SearchMatch]) -> int:
        """Calculate the width needed for line number alignment.

        Args:
            matches: List of match dictionaries.

        Returns:
            Width for line number display.
        """
        if not matches:
            return 3

        max_line = max(m["line"] for m in matches)
        return max(len(str(max_line)), 3)

    # =============================================================================
    # Output Truncation
    # =============================================================================

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
        truncated += "\n\n... [search results truncated after 80_000 characters. Use maxResults to limit output]"
        return truncated

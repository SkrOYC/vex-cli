"""Filesystem tools foundation for Vex CLI.

This package provides shared types, error handling, and services for filesystem tools.
"""

from __future__ import annotations

from vibe.core.tools.filesystem.create import CreateArgs, CreateResult, CreateTool
from vibe.core.tools.filesystem.edit import EditArgs, EditResult, EditTool
from vibe.core.tools.filesystem.edit_file import (
    EditFileArgs,
    EditFileResult,
    EditFileTool,
)
from vibe.core.tools.filesystem.grep import (
    GrepArgs,
    GrepResult,
    GrepTool,
)
from vibe.core.tools.filesystem.insert_line import (
    InsertLineArgs,
    InsertLineResult,
    InsertLineTool,
)
from vibe.core.tools.filesystem.list_files import (
    ListFilesArgs,
    ListFilesResult,
    ListFilesTool,
)
from vibe.core.tools.filesystem.read_file import (
    ReadFileArgs,
    ReadFileContentResult,
    ReadFileMediaResult,
    ReadFileResult,
    ReadFileTool,
)
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import (
    DEFAULT_CONTEXT_AFTER,
    DEFAULT_CONTEXT_BEFORE,
    DEFAULT_FILES_LIMIT,
    DEFAULT_RECURSIVE_DEPTH,
    FILE_EXISTS_ERROR_TIMEOUT_MS,
    LARGE_FILE_THRESHOLD,
    LENGTH_DIFF_RATIO_THRESHOLD,
    LINE_SIMILARITY_THRESHOLD,
    MAX_FILES_LIMIT,
    MAX_INLINE_MEDIA_BYTES,
    MAX_RECURSIVE_DEPTH,
    MIN_NEW_CONTENT_LENGTH,
    MIN_OLD_CONTENT_LENGTH,
    MISTAKEN_EDIT_TIMEOUT_MS,
    OUTPUT_LIMIT,
    STR_REPLACE_LENGTH_RATIO_THRESHOLD,
    FileSystemError,
    ValidationError,
)

__all__ = [
    "DEFAULT_CONTEXT_AFTER",
    "DEFAULT_CONTEXT_BEFORE",
    "DEFAULT_FILES_LIMIT",
    "DEFAULT_RECURSIVE_DEPTH",
    "FILE_EXISTS_ERROR_TIMEOUT_MS",
    "LARGE_FILE_THRESHOLD",
    "LENGTH_DIFF_RATIO_THRESHOLD",
    "LINE_SIMILARITY_THRESHOLD",
    "MAX_FILES_LIMIT",
    "MAX_INLINE_MEDIA_BYTES",
    "MAX_RECURSIVE_DEPTH",
    "MIN_NEW_CONTENT_LENGTH",
    "MIN_OLD_CONTENT_LENGTH",
    "MISTAKEN_EDIT_TIMEOUT_MS",
    "OUTPUT_LIMIT",
    "STR_REPLACE_LENGTH_RATIO_THRESHOLD",
    "CreateArgs",
    "CreateResult",
    "CreateTool",
    "EditArgs",
    "EditFileArgs",
    "EditFileResult",
    "EditFileTool",
    "EditResult",
    "EditTool",
    "FileSystemError",
    "GrepArgs",
    "GrepResult",
    "GrepTool",
    "InsertLineArgs",
    "InsertLineResult",
    "InsertLineTool",
    "ListFilesArgs",
    "ListFilesResult",
    "ListFilesTool",
    "ReadFileArgs",
    "ReadFileContentResult",
    "ReadFileMediaResult",
    "ReadFileResult",
    "ReadFileTool",
    "ValidationError",
    "ViewTrackerService",
]

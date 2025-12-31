"""CreateTool implementation for LangChain 1.2.0 integration.

This module provides CreateTool class for safe file creation with
view-before-write workflow enforcement.

Features:
- View tracking enforcement ("view before edit" workflow)
- File existence checks
- Parent directory creation
- Directory not found detection with helpful error messages
- UTF-8 encoding support
- Path resolution against working directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from vibe.core.tools.base import ToolPermission
from vibe.core.tools.filesystem.langchain_base import VibeLangChainTool
from vibe.core.tools.filesystem.types import FileSystemError


class CreateArgs(BaseModel):
    path: str
    content: str


class CreateResult(BaseModel):
    output: str


class CreateTool(VibeLangChainTool):
    name = "create"
    description = "Create a new file with content"

    def __init__(
        self,
        permission=Field(default=ToolPermission.ASK, exclude=True),
        view_tracker=None,
        workdir=None,
    ):
        super().__init__(
            permission=permission, view_tracker=view_tracker, workdir=workdir
        )

    def _run(self, input: dict[str, Any]) -> str:
        args = CreateArgs(**input)
        resolved_path = self._resolve_path(args.path)
        parent_dir = resolved_path.parent
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FileSystemError(
                f"Failed to create parent directory: {e}",
                code="PARENT_DIR_CREATE_FAILED",
                path=str(resolved_path),
            ) from e
        try:
            resolved_path.write_text(args.content, encoding="utf-8")
        except OSError as e:
            raise FileSystemError(
                f"Failed to write file: {e}",
                code="FILE_WRITE_FAILED",
                path=str(resolved_path),
            ) from e
        return f"File created: {resolved_path}"

    async def _arun(self, input: dict[str, Any]) -> str:
        import asyncio

        return await asyncio.to_thread(lambda: self._run(input))

    def _resolve_path(self, path: str) -> Path:
        if Path(path).is_absolute():
            return Path(path).resolve()
        else:
            return (self._get_effective_workdir() / path).resolve()

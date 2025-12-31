
"""InsertLineTool implementation for LangChain 1.2.0 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from vibe.core.tools.filesystem.langchain_base import VibeLangChainTool
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError

class InsertLineArgs(BaseModel):
    path: str
    line: int
    content: str

class InsertLineResult(BaseModel):
    output: str

class InsertLineTool(VibeLangChainTool):
    name = "insert_line"
    description = "Insert a line at specific position in a file"
    
    def __init__(self, permission=Field(default=ToolPermission.ASK, exclude=True), view_tracker=None, workdir=None):
        super().__init__(permission=permission, view_tracker=view_tracker, workdir=workdir)
    
    def _run(self, input: dict[str, Any]) -> str:
        args = InsertLineArgs(**input)
        resolved_path = self._resolve_path(args.path)
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
        if args.line < 0:
            lines.insert(args.line, args.content)
        else:
            lines.append(args.content)
        resolved_path.write_text("\n".join(lines), encoding="utf-8")
        return f"Line inserted at position {args.line} in {resolved_path}"
    
    async def _arun(self, input: dict[str, Any]) -> str:
        import asyncio
        return await asyncio.to_thread(lambda: self._run(input))
    
    def _resolve_path(self, path: str) -> Path:
        if Path(path).is_absolute():
            return Path(path).resolve()
        else:
            return (self._get_effective_workdir() / path).resolve()

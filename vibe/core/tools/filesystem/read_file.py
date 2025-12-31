
"""ReadFileTool implementation for LangChain 1.2.0 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from vibe.core.tools.filesystem.langchain_base import VibeLangChainTool
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError

class ReadFileArgs(BaseModel):
    path: str

class ReadFileResult(BaseModel):
    output: str

class ReadFileTool(VibeLangChainTool):
    name = "read_file"
    description = "Read file contents"
    
    def __init__(self, permission=Field(default=ToolPermission.ASK, exclude=True), view_tracker=None, workdir=None):
        super().__init__(permission=permission, view_tracker=view_tracker, workdir=workdir)
    
    def _run(self, input: dict[str, Any]) -> str:
        args = ReadFileArgs(**input)
        resolved_path = self._resolve_path(args.path)
        content = resolved_path.read_text(encoding="utf-8")
        if self.view_tracker:
            self.view_tracker.record_view(str(resolved_path))
        return content
    
    async def _arun(self, input: dict[str, Any]) -> str:
        import asyncio
        return await asyncio.to_thread(lambda: self._run(input))
    
    def _resolve_path(self, path: str) -> Path:
        if Path(path).is_absolute():
            return Path(path).resolve()
        else:
            return (self._get_effective_workdir() / path).resolve()

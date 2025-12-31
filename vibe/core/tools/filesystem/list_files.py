
"""ListFilesTool implementation for LangChain 1.2.0 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from vibe.core.tools.filesystem.langchain_base import VibeLangChainTool

class ListFilesArgs(BaseModel):
    path: str

class ListFilesResult(BaseModel):
    output: str

class ListFilesTool(VibeLangChainTool):
    name = "list_files"
    description = "List files in a directory"
    
    def __init__(self, permission=Field(default=ToolPermission.ASK, exclude=True), workdir=None):
        super().__init__(permission=permission, workdir=workdir)
    
    def _run(self, input: dict[str, Any]) -> str:
        args = ListFilesArgs(**input)
        resolved_path = self._resolve_path(args.path)
        if resolved_path.is_file():
            files = [resolved_path.name]
        else:
            files = [f.name for f in resolved_path.iterdir()]
        return "\n".join(files)
    
    async def _arun(self, input: dict[str, Any]) -> str:
        import asyncio
        return await asyncio.to_thread(lambda: self._run(input))
    
    def _resolve_path(self, path: str) -> Path:
        if Path(path).is_absolute():
            return Path(path).resolve()
        else:
            return (self._get_effective_workdir() / path).resolve()

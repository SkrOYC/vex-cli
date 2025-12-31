
"""GrepTool implementation for LangChain 1.2.0 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from vibe.core.tools.filesystem.langchain_base import VibeLangChainTool
from vibe.core.tools.filesystem.types import FileSystemError

class GrepArgs(BaseModel):
    path: str
    pattern: str

class GrepResult(BaseModel):
    output: str

class GrepTool(VibeLangChainTool):
    name = "grep"
    description = "Search for text in files"
    
    def __init__(self, permission=Field(default=ToolPermission.ASK, exclude=True), workdir=None):
        super().__init__(permission=permission, workdir=workdir)
    
    def _run(self, input: dict[str, Any]) -> str:
        args = GrepArgs(**input)
        resolved_path = self._resolve_path(args.path)
        if resolved_path.is_file():
            content = resolved_path.read_text(encoding="utf-8")
        else:
            content = ""
        import re
        matches = [line for line in content.splitlines() if re.search(args.pattern, line)]
        return "\n".join(matches)
    
    async def _arun(self, input: dict[str, Any]) -> str:
        import asyncio
        return await asyncio.to_thread(lambda: self._run(input))
    
    def _resolve_path(self, path: str) -> Path:
        if Path(path).is_absolute():
            return Path(path).resolve()
        else:
            return (self._get_effective_workdir() / path).resolve()

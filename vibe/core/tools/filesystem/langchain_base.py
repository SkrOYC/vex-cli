"""LangChain BaseTool with Vibe-specific features.

This module provides a base class that combines LangChain's BaseTool
with Vibe-specific features like permission checking and view tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from vibe.core.tools.base import ToolPermission


class VibeLangChainTool(BaseTool):
    """LangChain BaseTool with Vibe-specific features.

    This base class provides:
    - Permission checking (ALWAYS, NEVER, ASK)
    - View tracking integration
    - Effective workdir handling

    Note: Tool name is handled by LangChain's BaseTool class variable.
    """

    def __init__(
        self,
        permission: ToolPermission = ToolPermission.ASK,
        view_tracker: Any | None = None,
        workdir: Path | None = None,
    ) -> None:
        """Initialize VibeLangChainTool.

        Args:
            permission: Permission level for tool execution.
            view_tracker: Optional ViewTrackerService for tools that need it.
            workdir: Working directory for file operations.
        """
        self.permission = permission
        self.view_tracker = view_tracker
        self.workdir = workdir or Path.cwd()

    def _check_permission(self) -> bool:
        """Check if tool execution is allowed based on permission setting.

        Returns:
            True if execution is allowed, False otherwise.
        """
        if self.permission == ToolPermission.ALWAYS:
            return True
        if self.permission == ToolPermission.NEVER:
            return False
        return True  # ASK handled by middleware

    def _get_effective_workdir(self) -> Path:
        """Get effective working directory.

        Returns:
            Workdir from config or current working directory.
        """
        return self.workdir

    def _to_string_output(self, value: str | BaseModel | Any) -> str:
        """Convert value to string for LangChain compatibility.

        Args:
            value: Result to convert.

        Returns:
            String representation suitable for LangChain tool output.
        """
        if isinstance(value, BaseModel):
            if hasattr(value, "output"):
                return value.output
            else:
                return str(value)
        return str(value)

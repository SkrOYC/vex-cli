from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ToolError(RuntimeError):
    """Custom exception for tool errors."""

    pass


class ToolPermissionError(Exception):
    """Raised when a tool permission is not allowed."""


class ToolPermission:
    ALWAYS = "always"
    NEVER = "never"
    ASK = "ask"

    @classmethod
    def by_name(cls, name: str) -> str:
        name = name.lower()
        if name in {cls.ALWAYS, cls.NEVER, cls.ASK}:
            return name
        raise ToolPermissionError(
            f"Invalid tool permission: {name}. Must be one of {', '.join(map(repr, [cls.ALWAYS, cls.NEVER, cls.ASK]))}"
        )


class BaseToolConfig(BaseModel):
    """Configuration for a tool.

    Attributes:
        permission: The permission level required to use the tool.
        workdir: The working directory for the tool. If None, the current working directory is used.
        allowlist: Patterns that automatically allow tool execution.
        denylist: Patterns that automatically deny tool execution.
    """

    model_config = ConfigDict(extra="allow")

    permission: str = ToolPermission.ASK
    workdir: Path | None = Field(default=None, exclude=True)
    allowlist: list[str] = Field(default_factory=list)
    denylist: list[str] = Field(default_factory=list)

    @field_validator("workdir", mode="before")
    @classmethod
    def _expand_workdir(cls, v: Any) -> Path | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return None

    @property
    def effective_workdir(self) -> Path:
        return self.workdir if self.workdir is not None else Path.cwd()


class BaseToolState(BaseModel):
    """Base state for tools."""

    model_config = ConfigDict(extra="forbid")


# Simple base class for existing tools - will be removed in future migrations
class BaseTool(LangChainBaseTool):
    """Simple base class that inherits from LangChain BaseTool."""

    pass


class ToolUtils:
    """Utility functions for tool implementation."""

    @staticmethod
    def get_name(tool_class: type) -> str:
        """Convert CamelCase class name to snake_case tool name."""
        name = tool_class.__name__
        snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        return snake_case.replace("_tool", "")

    @staticmethod
    def resolve_path(path: str, workdir: Path) -> Path:
        """Resolve a path to absolute path within workdir."""
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = workdir / resolved
        return resolved.resolve()

    @staticmethod
    def validate_args(args: dict[str, Any], args_model: type[BaseModel]) -> BaseModel:
        """Validate arguments against the args model."""
        try:
            return args_model.model_validate(args)
        except ValidationError as err:
            raise ToolError(f"Validation error: {err}") from err

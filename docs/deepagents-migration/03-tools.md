# 03 - Tool System Migration

## Overview

Migrate from Mistral Vibe's complex `BaseTool[Args, Result, Config, State]` pattern to DeepAgents' simpler `StructuredTool` API while preserving tool functionality and TUI integration.

## Current Tool Architecture

### Complex Generic Pattern

```python
# vibe/core/tools/base.py - 284 lines
class BaseTool[
    ToolArgs: BaseModel,
    ToolResult: BaseModel,
    ToolConfig: BaseToolConfig,
    ToolState: BaseToolState,
](ABC):
    description: ClassVar[str]
    
    @abstractmethod
    async def run(self, args: ToolArgs) -> ToolResult: ...
    
    @classmethod
    def _get_tool_config_class(cls) -> type[ToolConfig]: ...
    
    @classmethod
    def _get_tool_state_class(cls) -> type[ToolState]: ...
    
    @classmethod
    def _get_tool_args_results(cls) -> tuple[type[ToolArgs], type[ToolResult]]: ...
    
    def check_allowlist_denylist(self, args: ToolArgs) -> ToolPermission | None: ...
```

### Current Tools

| Tool | Args | Result | Config | State |
|------|------|--------|--------|-------|
| `bash` | `BashArgs` | `BashResult` | `BashToolConfig` | `BaseToolState` |
| `read_file` | `ReadFileArgs` | `ReadFileResult` | `ReadFileToolConfig` | `ReadFileState` |
| `write_file` | `WriteFileArgs` | `WriteFileResult` | `WriteFileConfig` | `WriteFileState` |
| `search_replace` | `SearchReplaceArgs` | `SearchReplaceResult` | `SearchReplaceConfig` | `SearchReplaceState` |
| `todo` | `TodoArgs` | `TodoResult` | `TodoConfig` | `TodoState` |
| `grep` | `GrepArgs` | `GrepResult` | `GrepToolConfig` | `GrepState` |

## DeepAgents Tool Architecture

### Simple Function-Based Tools

```python
# DeepAgents approach - much simpler
from langchain_core.tools import StructuredTool

def read_file(file_path: str, offset: int = 0, limit: int = 500) -> str:
    """Read a file from the filesystem."""
    # Implementation
    return content

read_file_tool = StructuredTool.from_function(
    name="read_file",
    description="Read a file from the filesystem.",
    func=read_file,
)
```

### FilesystemMiddleware Provides Built-in Tools

DeepAgents' `FilesystemMiddleware` already provides:
- `ls` - List directory contents
- `read_file` - Read file with pagination
- `write_file` - Write new files
- `edit_file` - Edit existing files (search/replace)
- `glob` - Find files by pattern
- `grep` - Search file contents
- `execute` - Run shell commands (with SandboxBackend)

## Migration Strategy

### Option A: Use DeepAgents Tools Directly (Recommended)

Replace Mistral Vibe tools with DeepAgents equivalents:

| Mistral Vibe Tool | DeepAgents Equivalent | Notes |
|-------------------|----------------------|-------|
| `bash` | `execute` | Requires SandboxBackend or custom tool |
| `read_file` | `read_file` | Direct replacement |
| `write_file` | `write_file` | Direct replacement |
| `search_replace` | `edit_file` | Similar functionality |
| `todo` | `write_todos`/`read_todos` | Via TodoListMiddleware |
| `grep` | `grep` | Direct replacement |

### Option B: Wrap Existing Tools (If Custom Logic Needed)

```python
# vibe/core/engine/tools.py

from langchain_core.tools import StructuredTool
from vibe.core.tools.builtins.bash import Bash, BashArgs

def create_bash_tool(config: VibeConfig) -> StructuredTool:
    """Wrap Vibe's Bash tool for DeepAgents compatibility."""
    
    # Get the existing tool instance
    bash_instance = Bash(config=config.get_tool_config("bash"))
    
    async def bash_wrapper(
        command: str,
        workdir: str | None = None,
        timeout: int = 120,
    ) -> str:
        args = BashArgs(command=command, workdir=workdir, timeout=timeout)
        result = await bash_instance.run(args)
        return f"Exit code: {result.exit_code}\n{result.output}"
    
    return StructuredTool.from_function(
        name="bash",
        description=Bash.description,
        func=bash_wrapper,
        coroutine=bash_wrapper,
    )
```

## Implementation Plan

### Phase 1: Tool Adapter Module

Create `vibe/core/engine/tools.py`:

```python
"""Tool adapters for DeepAgents integration."""

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from vibe.core.config import VibeConfig


class VibeToolAdapter:
    """Adapts Vibe tools for DeepAgents consumption."""

    @staticmethod
    def get_all_tools(config: VibeConfig) -> Sequence[BaseTool]:
        """Get all tools configured for the agent."""
        tools: list[BaseTool] = []
        
        # Add custom bash tool (DeepAgents execute requires SandboxBackend)
        tools.append(VibeToolAdapter._create_bash_tool(config))
        
        # DeepAgents FilesystemMiddleware handles:
        # - read_file, write_file, edit_file, ls, glob, grep
        # These are added automatically by FilesystemMiddleware
        
        # DeepAgents TodoListMiddleware handles:
        # - write_todos, read_todos
        # These are added automatically by TodoListMiddleware
        
        # Add any custom tools from config
        for tool_path in config.tool_paths:
            custom_tools = VibeToolAdapter._load_custom_tools(tool_path)
            tools.extend(custom_tools)
        
        return tools

    @staticmethod
    def _create_bash_tool(config: VibeConfig) -> StructuredTool:
        """Create bash execution tool."""
        import asyncio
        import subprocess
        
        async def execute_bash(
            command: str,
            workdir: str | None = None,
            timeout: int = 120,
        ) -> str:
            """Execute a bash command."""
            cwd = workdir or str(config.effective_workdir)
            
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                output = stdout.decode("utf-8", errors="replace")
                return f"Exit code: {proc.returncode}\n{output}"
            except asyncio.TimeoutError:
                return f"Command timed out after {timeout} seconds"
            except Exception as e:
                return f"Error executing command: {e}"
        
        return StructuredTool.from_function(
            name="bash",
            description=(
                "Execute a bash command. Use for running scripts, "
                "git commands, package managers, etc."
            ),
            func=lambda *args, **kwargs: asyncio.run(execute_bash(*args, **kwargs)),
            coroutine=execute_bash,
        )

    @staticmethod
    def _load_custom_tools(tool_path: str) -> list[BaseTool]:
        """Load custom tools from a path."""
        # Preserve existing custom tool loading logic
        # but adapt to return LangChain BaseTool instances
        return []
```

### Phase 2: TUI Display Integration

The TUI needs tool call/result information for display. Create event handlers:

```python
# vibe/core/engine/adapters.py

from dataclasses import dataclass
from typing import Any

from vibe.core.types import ToolCallEvent, ToolResultEvent


@dataclass
class ToolCallDisplay:
    """Data for TUI tool call display."""
    tool_name: str
    tool_args: dict[str, Any]
    tool_id: str


@dataclass
class ToolResultDisplay:
    """Data for TUI tool result display."""
    tool_name: str
    tool_id: str
    result: str
    success: bool


class EventTranslator:
    """Translate DeepAgents events to Vibe TUI events."""

    @staticmethod
    def translate(event: dict[str, Any]) -> Any:
        """Translate a LangGraph event to Vibe event."""
        event_type = event.get("event")
        
        if event_type == "on_tool_start":
            return ToolCallDisplay(
                tool_name=event["name"],
                tool_args=event["data"].get("input", {}),
                tool_id=event["run_id"],
            )
        
        if event_type == "on_tool_end":
            return ToolResultDisplay(
                tool_name=event["name"],
                tool_id=event["run_id"],
                result=str(event["data"].get("output", "")),
                success=True,
            )
        
        if event_type == "on_chat_model_stream":
            # Handle streaming tokens
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content"):
                return {"type": "token", "content": chunk.content}
        
        return None
```

### Phase 3: Permission System Mapping

Map Vibe's permission system to DeepAgents HITL:

```python
# vibe/core/engine/permissions.py

from langchain.agents.middleware import InterruptOnConfig

from vibe.core.config import VibeConfig
from vibe.core.tools.base import ToolPermission


def build_interrupt_config(config: VibeConfig) -> dict[str, bool | InterruptOnConfig]:
    """Build DeepAgents interrupt config from Vibe tool permissions."""
    interrupt_on = {}
    
    # Default tools that need approval
    default_approval_tools = {
        "bash": True,
        "execute": True,
        "write_file": True,
        "edit_file": True,
    }
    
    # Start with defaults
    interrupt_on.update(default_approval_tools)
    
    # Override with config
    for tool_name, tool_config in config.tools.items():
        match tool_config.permission:
            case ToolPermission.ALWAYS:
                interrupt_on.pop(tool_name, None)  # No approval needed
            case ToolPermission.ASK:
                interrupt_on[tool_name] = True
            case ToolPermission.NEVER:
                # Tool is disabled - handled separately
                pass
    
    return interrupt_on
```

## Files to Remove After Migration

Once tools are fully migrated:

```
vibe/core/tools/
├── base.py           # Remove - replaced by StructuredTool
├── manager.py        # Simplify significantly
├── ui.py             # Preserve - needed for TUI display
├── mcp.py            # Preserve - MCP integration still useful
└── builtins/
    ├── bash.py       # Remove - replaced by simple function
    ├── read_file.py  # Remove - FilesystemMiddleware provides
    ├── write_file.py # Remove - FilesystemMiddleware provides
    ├── search_replace.py  # Remove - edit_file provides
    ├── todo.py       # Remove - TodoListMiddleware provides
    └── grep.py       # Remove - FilesystemMiddleware provides
```

## Validation Checklist

- [ ] Bash commands execute correctly
- [ ] File reading works with pagination
- [ ] File writing creates new files
- [ ] File editing (search/replace) works
- [ ] Todo management functional
- [ ] Grep searches return results
- [ ] Custom tools load from paths
- [ ] MCP tools still work
- [ ] TUI displays tool calls correctly
- [ ] TUI displays tool results correctly
- [ ] Permissions/approvals work correctly

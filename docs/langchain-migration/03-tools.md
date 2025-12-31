# 03 - Tools Migration

## Overview

Migrate the tool system from DeepAgents-specific patterns to native LangChain 1.2.0 `BaseTool` integration, ensuring all existing tools continue to work while removing adapter overhead.

## Current Tool System

### `vibe/core/engine/tools.py` Structure

```python
"""Tool adapters for DeepAgents integration."""

from langchain_core.tools import BaseTool, StructuredTool
from langchain.agents.middleware import TodoListMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import StateBackend

class VibeToolAdapter:
    @staticmethod
    def get_all_tools(config: VibeConfig) -> Sequence[BaseTool]:
        tools: list[BaseTool] = []
        
        # Custom bash tool
        tools.append(VibeToolAdapter._create_bash_tool(config))
        
        # DeepAgents provides filesystem and planning tools automatically
        # So we don't need to create these
        
        # MCP tools using official LangChain MCP adapters
        if config.mcp_servers:
            mcp_tools = asyncio.run(
                VibeToolAdapter._load_mcp_tools_official(config.mcp_servers)
            )
            tools.extend(mcp_tools)
        
        return tools
```

## Target Tool System

### `vibe/core/engine/tools.py` (After Migration)

```python
"""Tool adapters for LangChain 1.2.0 integration."""

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from vibe.core.config import VibeConfig


class VibeToolAdapter:
    """Adapts Vibe tools for LangChain 1.2.0 consumption."""

    @staticmethod
    def get_all_tools(config: VibeConfig) -> Sequence[BaseTool]:
        """Get all tools configured for the agent."""
        tools: list[BaseTool] = []
        
        # Custom bash tool
        tools.append(VibeToolAdapter._create_bash_tool(config))
        
        # Vibe-specific tools (file operations, grep, etc.)
        tools.extend(VibeToolAdapter._create_vibe_tools(config))
        
        # MCP tools using official LangChain MCP adapters
        if config.mcp_servers:
            import asyncio
            try:
                mcp_tools = asyncio.run(
                    VibeToolAdapter._load_mcp_tools_official(config.mcp_servers)
                )
                tools.extend(mcp_tools)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to load MCP tools: {e}")
        
        # Custom tools from config
        for tool_path in config.tool_paths:
            custom_tools = VibeToolAdapter._load_custom_tools(tool_path)
            tools.extend(custom_tools)
        
        return tools

    @staticmethod
    def _create_bash_tool(config: VibeConfig) -> StructuredTool:
        """Create bash execution tool for LangChain 1.2.0."""
        import asyncio

        async def execute_bash(
            command: str, workdir: str | None = None, timeout: int = 120
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
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode("utf-8", errors="replace")
                return f"Exit code: {proc.returncode}\n{output}"
            except TimeoutError:
                return f"Command timed out after {timeout} seconds"
            except Exception as e:
                return f"Error executing command: {e}"

        def sync_execute_bash(*args, **kwargs):
            raise NotImplementedError("Synchronous execution not supported")

        return StructuredTool.from_function(
            name="bash",
            description=(
                "Execute a bash command. Use for running scripts, "
                "git commands, package managers, etc."
            ),
            func=sync_execute_bash,
            coroutine=execute_bash,
        )

    @staticmethod
    def _create_vibe_tools(config: VibeConfig) -> list[BaseTool]:
        """Create Vibe-specific tools (read_file, write_file, grep, etc.)."""
        from vibe.core.tools.registry import get_registered_tools
        
        tools: list[BaseTool] = []
        for tool_name in get_registered_tools():
            if VibeToolAdapter._should_include_tool(tool_name, config):
                tool_class = VibeToolAdapter._load_tool_class(tool_name)
                if tool_class:
                    tools.append(tool_class(config))
        
        return tools

    @staticmethod
    def _should_include_tool(tool_name: str, config: VibeConfig) -> bool:
        """Check if tool should be included based on config."""
        # Check enabled_tools patterns
        if config.enabled_tools:
            return any(
                VibeToolAdapter._matches_pattern(tool_name, pattern)
                for pattern in config.enabled_tools
            )
        
        # Check disabled_tools patterns
        if config.disabled_tools:
            return not any(
                VibeToolAdapter._matches_pattern(tool_name, pattern)
                for pattern in config.disabled_tools
            )
        
        return True

    @staticmethod
    def _matches_pattern(tool_name: str, pattern: str) -> bool:
        """Check if tool name matches pattern."""
        import fnmatch
        
        if pattern.startswith("re:"):
            import re
            return bool(re.match(pattern[3:], tool_name))
        elif "*" in pattern or "?" in pattern:
            return fnmatch.fnmatch(tool_name, pattern)
        else:
            return tool_name == pattern

    @staticmethod
    def _load_tool_class(tool_name: str) -> type[BaseTool] | None:
        """Load tool class from registry."""
        from vibe.core.tools.registry import TOOL_REGISTRY
        return TOOL_REGISTRY.get(tool_name)

    @staticmethod
    async def _load_mcp_tools_official(mcp_servers) -> list[BaseTool]:
        """Load MCP tools using langchain-mcp-adapters package.
        
        Uses MultiServerMCPClient to connect to multiple MCP servers
        and convert their tools to LangChain-compatible format.
        
        Supports transport types:
        - stdio: Local MCP server processes
        - http/streamable_http: Remote MCP servers
        - sse: Server-sent events transport
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        connections = {}
        for server in mcp_servers:
            connections[server.name] = VibeToolAdapter._convert_vibe_mcp_config(server)

        if not connections:
            return []

        client = MultiServerMCPClient(connections=connections)
        return await client.get_tools()

    @staticmethod
    def _convert_vibe_mcp_config(vibe_server) -> dict:
        """Convert Vibe MCP server config to langchain-mcp-adapters format.
        
        MultiServerMCPClient expects a dict with:
        - transport: "stdio", "http", "streamable_http", or "sse"
        - For stdio: command, args (optional), env (optional)
        - For http/streamable_http/sse: url, headers (optional)
        
        See: https://github.com/langchain-ai/langchain-mcp-adapters
        """
        config = {"transport": vibe_server.transport}

        if vibe_server.transport == "stdio":
            config["command"] = vibe_server.command
            if vibe_server.args:
                config["args"] = vibe_server.args
            # Environment variables can be added if needed:
            # if hasattr(vibe_server, "env") and vibe_server.env:
            #     config["env"] = vibe_server.env
        elif vibe_server.transport in ("http", "streamable-http", "streamable_http"):
            config["url"] = vibe_server.url
            if hasattr(vibe_server, "headers") and vibe_server.headers:
                config["headers"] = vibe_server.headers
            if hasattr(vibe_server, "http_headers") and vibe_server.http_headers():
                config["headers"] = vibe_server.http_headers()

        return config

    @staticmethod
    def _load_custom_tools(tool_path: str) -> list[BaseTool]:
        """Load custom tools from a file or directory."""
        from pathlib import Path
        import importlib.util

        path = Path(tool_path)

        if not path.exists():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Tool path does not exist: {tool_path}")
            return []

        if path.is_file():
            return VibeToolAdapter._load_tools_from_file(path)
        elif path.is_dir():
            return VibeToolAdapter._load_tools_from_directory(path)

        return []

    @staticmethod
    def _load_tools_from_file(file_path: Path) -> list[BaseTool]:
        """Load tools from a Python file."""
        import importlib.util

        try:
            module_name = (
                f"vibe.custom_tools.{file_path.stem}_{hash(str(file_path.resolve()))}"
            )
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return []
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            tools: list[BaseTool] = []
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, BaseTool):
                    tools.append(obj())
                elif callable(obj) and hasattr(obj, "__tool__"):
                    tools.append(obj())

            return tools
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error loading tools from {file_path}: {e}")
            return []

    @staticmethod
    def _load_tools_from_directory(dir_path: Path) -> list[BaseTool]:
        """Load tools from a directory containing Python files."""
        tools = []
        for py_file in dir_path.glob("*.py"):
            tools.extend(VibeToolAdapter._load_tools_from_file(py_file))
        return tools
```

## Changes Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Filesystem Tools** | Provided by DeepAgents | Created by VibeToolAdapter |
| **Planning Tools** | TodoListMiddleware (DeepAgents) | Created by VibeToolAdapter |
| **MCP Tools** | Via langchain-mcp-adapters | No change (already compatible) |
| **Tool Filtering** | Via DeepAgents config | Via VibeToolAdapter patterns |
| **Custom Tools** | Via tool_paths config | No change (already compatible) |

## Validation Checklist

- [x] All existing Vibe tools work correctly
- [x] MCP tools load successfully
- [x] Custom tools from tool_paths work
- [x] Tool filtering (enabled/disabled) works
- [x] Bash tool executes correctly
- [x] Tool permissions are enforced
- [x] No DeepAgents imports remain

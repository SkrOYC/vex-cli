"""Tool adapters for LangChain 1.2.0 integration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Never

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from vibe.core.config import VibeConfig
from vibe.core.tools.filesystem import (
    CreateTool,
    EditTool,
    InsertLineTool,
    ReadFileTool,
    ListFilesTool,
    StrReplaceTool,
    GrepTool,
)
from vibe.core.tools.filesystem.shared import ViewTrackerService


def _should_include_tool(tool_name: str, config: VibeConfig) -> bool:
    """Check if tool should be included based on config filtering.

    Args:
        tool_name: The tool's name attribute.
        config: VibeConfig instance with enabled_tools and disabled_tools.

    Returns:
        True if tool should be included, False otherwise.
    """
    # Whitelist mode: only include tools in enabled_tools
    if config.enabled_tools:
        for pattern in config.enabled_tools:
            if _matches_tool_pattern(tool_name, pattern):
                return True
        return False

    # Blacklist mode: exclude tools in disabled_tools
    if config.disabled_tools:
        for pattern in config.disabled_tools:
            if _matches_tool_pattern(tool_name, pattern):
                return False

    # Default: include all tools
    return True


def _matches_tool_pattern(tool_name: str, pattern: str) -> bool:
    """Check if tool name matches pattern (exact, glob, or regex).

    Args:
        tool_name: The tool's name.
        pattern: Pattern to match (supports exact, glob, re: prefix).

    Returns:
        True if tool_name matches pattern.
    """
    import fnmatch
    import re

    if pattern.startswith("re:"):
        return bool(re.match(pattern[3:], tool_name))
    elif "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(tool_name, pattern)
    else:
        return tool_name == pattern


class VibeToolAdapter:
    """Adapts Vibe tools for LangChain 1.2.0 consumption."""

    @staticmethod
    def get_all_tools(config: VibeConfig) -> Sequence[BaseTool]:
        """Get all tools configured for agent.

        This method creates all available tools and applies filtering based on
        config.enabled_tools (whitelist) and config.disabled_tools (blacklist).

        Tool sources:
        - Bash tool (always included)
        - Filesystem tools (create, edit, str_replace, insert_line, read_file, list_files, grep)
        - MCP tools (if configured)
        - Custom tools (from tool_paths)

        Args:
            config: VibeConfig instance with tool filtering configuration.

        Returns:
            Sequence of LangChain-compatible BaseTool instances.
        """
        tools: list[BaseTool] = []

        # Create shared ViewTrackerService for tools that need it
        view_tracker = ViewTrackerService()

        # Add bash tool
        tools.append(VibeToolAdapter._create_bash_tool(config))

        # Add filesystem tools with filtering
        filesystem_tools = [
            CreateTool,
            EditTool,
            StrReplaceTool,
            InsertLineTool,
            ReadFileTool,
            ListFilesTool,
            GrepTool,
        ]

        for tool_class in filesystem_tools:
            if _should_include_tool(tool_class.name, config):
                # Instantiate tool with permission, view_tracker, and workdir
                from vibe.core.tools.base import ToolPermission

                tool_config_class = tool_class._get_tool_config_class()

                # Create config with view_tracker for tools that need it
                config_kwargs: dict[str, Any] = {
                    "permission": ToolPermission.ASK,
                    "workdir": config.effective_workdir,
                }
                if tool_class.name in (
                    "create",
                    "edit",
                    "str_replace",
                    "insert_line",
                    "read_file",
                ):
                    config_kwargs["view_tracker"] = view_tracker

                tool_config = tool_config_class(**config_kwargs)

                # Get state class and instantiate
                tool_state_class = tool_class._get_tool_state_class()
                tool_state = tool_state_class()

                # Instantiate tool
                tools.append(tool_class(config=tool_config, state=tool_state))

        # Add MCP tools using official LangChain MCP adapters
        if config.mcp_servers:
            import asyncio

            try:
                mcp_tools = asyncio.run(
                    VibeToolAdapter._load_mcp_tools_official(config.mcp_servers)
                )
                # Filter MCP tools
                for tool in mcp_tools:
                    if _should_include_tool(tool.name, config):
                        tools.append(tool)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Failed to load MCP tools: {e}")

        # Add any custom tools from config
        for tool_path in config.tool_paths:
            custom_tools = VibeToolAdapter._load_custom_tools(tool_path)
            # Filter custom tools
            for tool in custom_tools:
                if _should_include_tool(tool.name, config):
                    tools.append(tool)

        return tools

    @staticmethod
    def _create_bash_tool(config: VibeConfig) -> BaseTool:
        """Create bash execution tool."""
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

        def sync_execute_bash(*args: Any, **kwargs: Any) -> Never:
            raise NotImplementedError("Synchronous execution not supported")

        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            name="bash",
            description=(
                "Execute a bash command. Use for running scripts, "
                "git commands, package managers, etc."
            ),
            func=sync_execute_bash,
        )

    @staticmethod
    def _load_custom_tools(tool_path: str) -> list[BaseTool]:
        """Load custom tools from a file or directory."""
        from pathlib import Path

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
            # Create a unique module name to avoid collisions in sys.modules
            module_name = (
                f"vibe.custom_tools.{file_path.stem}_{hash(str(file_path.resolve()))}"
            )
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return []
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            tools = []
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, BaseTool):
                    tools.append(obj)
                elif callable(obj) and hasattr(obj, "__tool__"):
                    # Support @tool decorator pattern
                    tools.append(obj)

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

    @staticmethod
    async def _load_mcp_tools_official(mcp_servers: list[Any]) -> list[BaseTool]:
        """Load MCP tools using official LangChain MCP adapters."""
        connections = {}
        for server in mcp_servers:
            # Convert Vibe MCP config to langchain-mcp-adapters format
            connection_config = VibeToolAdapter._convert_vibe_mcp_config(server)
            connections[server.name] = connection_config

        if not connections:
            return []

        try:
            client = MultiServerMCPClient(connections=connections)
            return await client.get_tools()
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error loading MCP tools: {e}")
            return []

    @staticmethod
    def _convert_vibe_mcp_config(vibe_server: Any) -> dict:
        """Convert Vibe MCP server config to langchain-mcp-adapters format."""
        config = {"transport": vibe_server.transport}

        if vibe_server.transport == "stdio":
            config["command"] = vibe_server.command
            if vibe_server.args:
                config["args"] = vibe_server.args
        elif vibe_server.transport in {"http", "streamable-http", "streamable_http"}:
            config["url"] = vibe_server.url
            if hasattr(vibe_server, "headers") and vibe_server.headers:
                config["headers"] = vibe_server.headers
            # Handle API key if configured
            if hasattr(vibe_server, "http_headers") and vibe_server.http_headers():
                config["headers"] = vibe_server.http_headers()

        return config

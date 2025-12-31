"""Tool adapters for LangChain 1.2.0 integration.

This module provides tools adapted for use with LangChain-based agent engines.
It includes bash execution, filesystem operations (create, read, edit, list, grep),
MCP server tools, and custom tools from configured paths.
"""

from __future__ import annotations

from collections.abc import Sequence
import fnmatch
from pathlib import Path
import re
from typing import Any, Never

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from vibe.core.config import VibeConfig
from vibe.core.tools.filesystem.create import CreateTool
from vibe.core.tools.filesystem.edit import EditTool
from vibe.core.tools.filesystem.edit_file import EditFileTool
from vibe.core.tools.filesystem.grep import GrepTool
from vibe.core.tools.filesystem.insert_line import InsertLineTool
from vibe.core.tools.filesystem.list_files import ListFilesTool
from vibe.core.tools.filesystem.read_file import ReadFileTool


class VibeToolAdapter:
    """Adapts Vibe tools for LangChain consumption.

    This class provides tools adapted for use with LangChain-based agent engines.
    All tools can be filtered using the enabled_tools and disabled_tools configuration
    options to control which tools are available to the agent.
    """

    @staticmethod
    def _get_filesystem_tools(config: VibeConfig) -> list[BaseTool]:
        """Get filesystem tools for file operations.

        Args:
            config: VibeConfig containing working directory.

        Returns:
            List of filesystem BaseTool instances (create, read, edit, edit_file, list, grep, insert_line).
        """
        tools: list[BaseTool] = []
        workdir = config.effective_workdir

        # Create all filesystem tools
        tools.append(CreateTool(workdir=workdir))
        tools.append(ReadFileTool(workdir=workdir))
        tools.append(EditTool(workdir=workdir))
        tools.append(EditFileTool(workdir=workdir))
        tools.append(ListFilesTool(workdir=workdir))
        tools.append(GrepTool(workdir=workdir))
        tools.append(InsertLineTool(workdir=workdir))

        return tools

    @staticmethod
    def _matches_tool_pattern(tool_name: str, pattern: str) -> bool:
        """Check if tool name matches the given pattern.

        Supports:
        - Exact name match (e.g., "bash")
        - Glob patterns (e.g., "bash*", "file_*")
        - Regex patterns with re: prefix (e.g., "re:^file_.*")

        Args:
            tool_name: Name of the tool to check.
            pattern: Pattern to match against (exact, glob, or regex with re: prefix).

        Returns:
            True if tool name matches pattern, False otherwise.
        """
        if not pattern:
            return False

        # Regex pattern with re: prefix
        if pattern.startswith("re:"):
            regex_pattern = pattern[3:]
            try:
                return bool(re.match(regex_pattern, tool_name))
            except re.error:
                return False

        # Glob pattern (contains * or ?)
        if "*" in pattern or "?" in pattern:
            return fnmatch.fnmatch(tool_name, pattern)

        # Exact match
        return tool_name == pattern

    @staticmethod
    def _apply_tool_filtering(
        tools: list[BaseTool], config: VibeConfig
    ) -> list[BaseTool]:
        """Apply enabled_tools and disabled_tools filtering to tool list.

        Behavior:
        - If enabled_tools is set, only include tools matching those patterns (whitelist)
        - If disabled_tools is set, exclude tools matching those patterns (blacklist)
        - If both are set, disabled_tools takes precedence (blacklist overrides whitelist)
        - If neither is set, all tools are included

        Args:
            tools: List of tools to filter.
            config: VibeConfig containing enabled_tools and disabled_tools lists.

        Returns:
            Filtered list of tools based on config.
        """
        # If whitelist is set, filter to only matching tools
        if config.enabled_tools:
            tools = [
                tool
                for tool in tools
                if any(
                    VibeToolAdapter._matches_tool_pattern(tool.name, pattern)
                    for pattern in config.enabled_tools
                )
            ]

        # Apply blacklist (overrides whitelist if both are set)
        if config.disabled_tools:
            tools = [
                tool
                for tool in tools
                if not any(
                    VibeToolAdapter._matches_tool_pattern(tool.name, pattern)
                    for pattern in config.disabled_tools
                )
            ]

        return tools

    @staticmethod
    def get_all_tools(config: VibeConfig) -> Sequence[BaseTool]:
        """Get all tools configured for the agent.

        Args:
            config: VibeConfig containing tool configuration and filtering options.

        Returns:
            Sequence of BaseTool instances available to the agent.
            Tools are filtered based on enabled_tools and disabled_tools config.
        """
        tools: list[BaseTool] = []

        # Add custom bash tool
        tools.append(VibeToolAdapter._create_bash_tool(config))

        # Add filesystem tools (create, read, edit, list, grep)
        tools.extend(VibeToolAdapter._get_filesystem_tools(config))

        # Add MCP tools using official LangChain MCP adapters
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

        # Add any custom tools from config
        for tool_path in config.tool_paths:
            custom_tools = VibeToolAdapter._load_custom_tools(tool_path)
            tools.extend(custom_tools)

        # Apply tool filtering based on config
        return VibeToolAdapter._apply_tool_filtering(tools, config)

    @staticmethod
    def _create_bash_tool(config: VibeConfig) -> StructuredTool:
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

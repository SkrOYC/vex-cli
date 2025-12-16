"""Tool adapters for DeepAgents integration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from langchain.agents.middleware import TodoListMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import StateBackend

from vibe.core.config import VibeConfig


class VibeToolAdapter:
    """Adapts Vibe tools for DeepAgents consumption."""

    @staticmethod
    def get_all_tools(config: VibeConfig) -> Sequence[BaseTool]:
        """Get all tools configured for the agent."""
        tools: list[BaseTool] = []

        # Add custom bash tool
        tools.append(VibeToolAdapter._create_bash_tool(config))

        # DeepAgents provides filesystem and planning (TodoList) tools by default,
        # so we don't need to create these middleware instances here

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

        return tools

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
    async def _load_mcp_tools_official(mcp_servers) -> list[BaseTool]:
        """Load MCP tools using official LangChain MCP adapters."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

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
    def _convert_vibe_mcp_config(vibe_server) -> dict:
        """Convert Vibe MCP server config to langchain-mcp-adapters format."""
        config = {"transport": vibe_server.transport}

        if vibe_server.transport == "stdio":
            config["command"] = vibe_server.command
            if vibe_server.args:
                config["args"] = vibe_server.args
        elif vibe_server.transport in ("http", "streamable-http", "streamable_http"):
            config["url"] = vibe_server.url
            if hasattr(vibe_server, "headers") and vibe_server.headers:
                config["headers"] = vibe_server.headers
            # Handle API key if configured
            if hasattr(vibe_server, "http_headers") and vibe_server.http_headers():
                config["headers"] = vibe_server.http_headers()

        return config

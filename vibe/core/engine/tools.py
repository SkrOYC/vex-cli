"""Tool adapters for DeepAgents integration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from langchain.agents.middleware import TodoListMiddleware

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import StateBackend

from vibe.core.config import VibeConfig


class VibeToolAdapter:
    """Adapts Vibe tools for DeepAgents consumption."""

    @staticmethod
    def get_all_tools(config: VibeConfig) -> Sequence[BaseTool]:
        """Get all tools configured for the agent."""
        tools: list[BaseTool] = []

        # Add custom bash tool (DeepAgents execute requires SandboxBackend)
        tools.append(VibeToolAdapter._create_bash_tool(config))

        # Add filesystem tools from FilesystemMiddleware
        filesystem_middleware = FilesystemMiddleware(
            backend=lambda rt: StateBackend(rt)
        )
        tools.extend(filesystem_middleware.tools)

        # Add planning tools from TodoListMiddleware
        todo_middleware = TodoListMiddleware()
        tools.extend(todo_middleware.tools)

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
            spec = importlib.util.spec_from_file_location("custom_tools", file_path)
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

        tools = []

        try:
            if hasattr(mcp_server, "url"):  # HTTP server
                remote_tools = asyncio.run(list_tools_http(mcp_server))
                for remote_tool in remote_tools:
                    tool_class = create_mcp_http_proxy_tool_class(
                        remote_tool, mcp_server
                    )
                    # Adapt to LangChain BaseTool
                    adapted_tool = VibeToolAdapter._adapt_mcp_tool(tool_class)
                    tools.append(adapted_tool)
            elif hasattr(mcp_server, "command"):  # Stdio server
                remote_tools = asyncio.run(list_tools_stdio(mcp_server))
                for remote_tool in remote_tools:
                    tool_class = create_mcp_stdio_proxy_tool_class(
                        remote_tool, mcp_server
                    )
                    # Adapt to LangChain BaseTool
                    adapted_tool = VibeToolAdapter._adapt_mcp_tool(tool_class)
                    tools.append(adapted_tool)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error loading MCP tools from {mcp_server}: {e}")

        return tools

    @staticmethod
    def _adapt_mcp_tool(mcp_tool_class) -> BaseTool:
        """Adapt an MCP tool class to LangChain BaseTool."""
        # Instantiate the MCP tool
        mcp_tool_instance = mcp_tool_class()

        # Create a wrapper function
        async def mcp_wrapper(*args, **kwargs):
            # Call the MCP tool's run method
            result = await mcp_tool_instance.arun(*args, **kwargs)
            # Return the result (assuming it's compatible)
            return result

        # Create StructuredTool
        return StructuredTool.from_function(
            name=mcp_tool_instance.name,
            description=mcp_tool_instance.description or "",
            func=lambda *args, **kwargs: None,  # Sync not supported
            coroutine=mcp_wrapper,
        )

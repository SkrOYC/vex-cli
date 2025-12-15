"""Tool adapters for DeepAgents integration."""

from __future__ import annotations

from collections.abc import Sequence

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
            except TimeoutError:
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
"""Event and approval adapters for DeepAgents integration."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel
from vibe.core.config import VibeConfig
from vibe.core.tools.base import BaseTool
from vibe.core.tools.manager import ToolManager
from vibe.core.types import AssistantEvent, BaseEvent, ToolCallEvent, ToolResultEvent


class GenericArgs(BaseModel):
    """Generic args fallback."""
    args: dict = {}


class GenericResult(BaseModel):
    """Generic result fallback."""
    output: str = ""


class EventTranslator:
    """Translate LangGraph events to Vibe TUI events."""

    def __init__(self, config: VibeConfig):
        self.config = config
        self.tool_manager = ToolManager(config)

    def _get_tool_info(self, tool_name: str) -> tuple[type[BaseTool] | None, type[BaseModel] | None]:
        """Get tool class and args model for a tool name."""
        try:
            tool_instance = self.tool_manager.get(tool_name)
            tool_class = tool_instance.__class__
            args_model = tool_class._get_tool_args_results()[0]
            return tool_class, args_model
        except Exception:
            return None, None

    def translate(self, event: dict[str, Any]) -> BaseEvent | None:
        """Translate a single LangGraph event to Vibe event."""
        event_type = event.get("event")

        match event_type:
            case "on_chat_model_stream":
                # Streaming token
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    return AssistantEvent(content=chunk.content)

            case "on_tool_start":
                # Tool call starting
                tool_name = event.get("name", "")
                tool_args = event.get("data", {}).get("input", {})

                tool_class, args_model = self._get_tool_info(tool_name)
                if tool_class and args_model:
                    try:
                        args = args_model.model_validate(tool_args)
                    except Exception:
                        # Fallback to generic args if validation fails
                        args = GenericArgs(args=tool_args)
                else:
                    # Fallback
                    args = GenericArgs(args=tool_args)
                    tool_class = None

                return ToolCallEvent(
                    tool_name=tool_name,
                    tool_class=tool_class,  # type: ignore
                    args=args,
                    tool_call_id=event.get("run_id", "")
                )

            case "on_tool_end":
                # Tool call complete
                tool_name = event.get("name", "")
                tool_result = event.get("data", {}).get("output", "")

                tool_class, _ = self._get_tool_info(tool_name)
                if tool_class:
                    result_model = tool_class._get_tool_args_results()[1]
                    try:
                        result = result_model.model_validate({"output": tool_result})
                    except Exception:
                        # Fallback to generic result
                        result = GenericResult(output=str(tool_result))
                else:
                    # Fallback
                    result = GenericResult(output=str(tool_result))

                return ToolResultEvent(
                    tool_name=tool_name,
                    tool_class=tool_class,  # type: ignore
                    result=result,
                    tool_call_id=event.get("run_id", "")
                )

        return None


class ApprovalBridge:
    """Bridge LangGraph interrupts to Vibe TUI approval flow."""

    def __init__(self) -> None:
        self._pending_approval: asyncio.Future[dict[str, Any]] | None = None

    async def handle_interrupt(self, interrupt: dict[str, Any]) -> dict[str, Any]:
        """Handle LangGraph interrupts for approval."""
        # For skeleton implementation, return approved immediately
        # In real implementation, this would create a Future and wait for TUI response
        return {"approved": True}

    async def respond(self, approved: bool, feedback: str | None = None) -> None:
        """Respond to pending approval."""
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result({"approved": approved, "feedback": feedback})
            self._pending_approval = None
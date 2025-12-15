"""Event and approval adapters for DeepAgents integration."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from vibe.core.config import VibeConfig
from vibe.core.tools.base import BaseTool
from vibe.core.tools.manager import ToolManager
from vibe.core.types import (
    AssistantEvent,
    BaseEvent,
    InterruptEvent,
    ToolCallEvent,
    ToolResultEvent,
)


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

    def _get_tool_info(
        self, tool_name: str
    ) -> tuple[type[BaseTool] | None, type[BaseModel] | None]:
        """Get tool class and args model for a tool name."""
        try:
            tool_instance = self.tool_manager.get(tool_name)
            tool_class = tool_instance.__class__
            args_model = tool_class._get_tool_args_results()[0]
            return tool_class, args_model
        except Exception:
            return None, None

    def translate(self, event: dict[str, Any] | Any) -> BaseEvent | None:
        """Translate a single LangGraph event to Vibe TUI events."""
        # Handle both dict and StreamEvent objects
        if not isinstance(event, dict):
            event_data = event.__dict__ if hasattr(event, "__dict__") else {}
        else:
            event_data = event
        event_type = event_data.get("event")

        match event_type:
            case "on_chat_model_stream":
                # Streaming token
                chunk = event_data.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    return AssistantEvent(content=chunk.content)

            case "interrupt":
                # Agent interrupt requiring approval
                return InterruptEvent(interrupt_data=event_data)

            case "on_tool_start":
                # Tool call starting
                tool_name = event_data.get("name", "")
                tool_args = event_data.get("data", {}).get("input", {})

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
                    tool_call_id=event_data.get("run_id", ""),
                )

            case "on_tool_end":
                # Tool call complete
                tool_name = event_data.get("name", "")
                tool_result = event_data.get("data", {}).get("output", "")

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
                    tool_call_id=event_data.get("run_id", ""),
                )

        return None


class ApprovalBridge:
    """Bridge LangGraph interrupts to Vibe TUI approval flow."""

    def __init__(self) -> None:
        self._pending_approvals: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def handle_interrupt(self, interrupt: dict[str, Any]) -> dict[str, Any]:
        """Handle LangGraph interrupts for approval."""
        # Extract ActionRequest from interrupt data
        action_request = self._extract_action_request(interrupt)
        if not action_request:
            return {"approved": True}  # Auto-approve if no action request

        # Create unique request ID
        request_id = str(uuid4())

        # Create Future for approval decision
        future = asyncio.Future()
        self._pending_approvals[request_id] = future

        # In real implementation, this would trigger TUI dialog via callback
        # For now, auto-approve (will be connected to TUI later)
        try:
            # Wait for decision with timeout (1 second for testing)
            decision = await asyncio.wait_for(future, timeout=1.0)
            return decision
        except asyncio.TimeoutError:
            # Auto-reject on timeout
            if request_id in self._pending_approvals:
                del self._pending_approvals[request_id]
            return {"approved": False, "feedback": "Approval timeout"}
        finally:
            # Clean up
            if request_id in self._pending_approvals:
                del self._pending_approvals[request_id]

    def _extract_action_request(
        self, interrupt: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract ActionRequest from LangGraph interrupt data."""
        # Based on the interrupt structure from docs
        data = interrupt.get("data")
        if data is not None:
            action_request = data.get("action_request")
            if action_request and isinstance(action_request, dict):
                # Ensure all required fields are present with defaults
                return {
                    "name": action_request.get("name", ""),
                    "args": action_request.get("args", {}),
                    "description": action_request.get("description", ""),
                }

        # Fallback: try to extract from other possible structures
        if "name" in interrupt and "args" in interrupt:
            return {
                "name": interrupt["name"],
                "args": interrupt.get("args", {}),
                "description": interrupt.get("description", ""),
            }

        return None

    async def respond(
        self, approved: bool, request_id: str, feedback: str | None = None
    ) -> None:
        """Respond to pending approval."""
        if request_id in self._pending_approvals:
            future = self._pending_approvals[request_id]
            if not future.done():
                future.set_result({"approved": approved, "feedback": feedback})
            del self._pending_approvals[request_id]

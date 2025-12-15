"""Event and approval adapters for DeepAgents integration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from vibe.core.tools.base import BaseTool
from vibe.core.types import BaseEvent


class DummyTool(BaseTool):
    """Dummy tool for testing."""
    description = "Dummy tool"

    def __init__(self):
        super().__init__(None, None)  # type: ignore

    async def run(self, args):
        return None


class EventTranslator:
    """Translate LangGraph events to Vibe TUI events."""

    @staticmethod
    def translate(event: dict[str, Any]) -> BaseEvent | None:
        """Translate a single LangGraph event to Vibe event."""
        event_type = event.get("event")
        
        match event_type:
            case "on_chat_model_stream":
                # Streaming token
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    from vibe.core.types import AssistantEvent
                    return AssistantEvent(content=chunk.content)
            
            case "on_tool_start":
                # Tool call starting
                tool_name = event.get("name", "")
                tool_args = event.get("data", {}).get("input", {})

                from vibe.core.types import ToolCallEvent
                from pydantic import BaseModel

                class DummyArgs(BaseModel):
                    args: dict = {}

                return ToolCallEvent(
                    tool_name=tool_name,
                    tool_class=DummyTool,
                    args=DummyArgs(args=tool_args),
                    tool_call_id=event.get("run_id", "")
                )

            case "on_tool_end":
                # Tool call complete
                tool_name = event.get("name", "")
                tool_result = event.get("data", {}).get("output", "")

                from vibe.core.types import ToolResultEvent
                from pydantic import BaseModel

                class DummyResult(BaseModel):
                    output: str = ""

                return ToolResultEvent(
                    tool_name=tool_name,
                    tool_class=DummyTool,
                    result=DummyResult(output=str(tool_result)),
                    tool_call_id=event.get("run_id", "")
                )
        
        return None


class ApprovalBridge:
    """Bridge LangGraph interrupts to Vibe TUI approval flow."""
    
    def __init__(self) -> None:
        self._pending_approval = None
    
    async def handle_interrupt(self, interrupt: dict[str, Any]) -> dict[str, Any]:
        """Handle LangGraph interrupts for approval."""
        # Placeholder - will integrate with TUI approval dialogs
        return {"approved": True}
    
    async def respond(self, approved: bool, feedback: str | None = None) -> None:
        """Respond to pending approval."""
        if self._pending_approval:
            self._pending_approval.set_result({"approved": approved, "feedback": feedback})
            self._pending_approval = None
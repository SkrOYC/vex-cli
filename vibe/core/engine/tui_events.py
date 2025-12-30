"""Map native LangGraph events to Vibe TUI events.

This module provides the TUIEventMapper class that translates native LangGraph
events from VibeLangChainEngine's astream_events() output to Vibe TUI event
types that EventHandler expects.

This is part of the migration from DeepAgents to native LangChain 1.2.0,
replacing the EventTranslator adapter layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from vibe.core.tools.manager import ToolManager
from vibe.core.types import AssistantEvent, BaseEvent, ToolCallEvent, ToolResultEvent
from vibe.core.utils import logger

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig
    from vibe.core.tools.base import BaseTool


class GenericArgs(BaseModel):
    """Generic args fallback for tool calls without specific model."""

    args: dict = {}


class GenericResult(BaseModel):
    """Generic result fallback for tool results without specific model."""

    output: str = ""


class TUIEventMapper:
    """Map native LangGraph events to Vibe TUI events.

    This mapper translates events from VibeLangChainEngine's native
    astream_events() output to Vibe TUI event types that
    EventHandler expects.

    Supported event types:
    - on_chat_model_stream -> AssistantEvent (token streaming)
    - on_tool_start -> ToolCallEvent (tool execution begins)
    - on_tool_end -> ToolResultEvent (tool execution completes)

    Example:
        mapper = TUIEventMapper(config)
        async for event in engine.run(message):
            mapped = mapper.map_event(event)
            if mapped:
                await handler.handle_event(mapped)
    """

    def __init__(self, config: VibeConfig) -> None:
        """Initialize the event mapper with configuration.

        Args:
            config: VibeConfig containing tool definitions and permissions
        """
        self.config = config
        self.tool_manager = ToolManager(config)

    def map_event(self, event: dict[str, Any] | Any) -> BaseEvent | None:
        """Map a native LangGraph event to a Vibe TUI event.

        Args:
            event: Either a dict (from astream_events) or a StreamEvent object

        Returns:
            A Vibe TUI event (AssistantEvent, ToolCallEvent, ToolResultEvent)
            or None if the event should be ignored
        """
        # Handle both dict and object types from astream_events
        if not isinstance(event, dict):
            event_data = event.__dict__ if hasattr(event, "__dict__") else {}
        else:
            event_data = event

        event_type = event_data.get("event", "")

        match event_type:
            case "on_chat_model_stream":
                return self._map_chat_model_stream(event_data)

            case "on_tool_start":
                return self._map_tool_start(event_data)

            case "on_tool_end":
                return self._map_tool_end(event_data)

            # Other event types are not relevant for TUI display
            case _:
                return None

    def _map_chat_model_stream(
        self, event_data: dict[str, Any]
    ) -> AssistantEvent | None:
        """Map on_chat_model_stream event to AssistantEvent.

        Args:
            event_data: The event data dict containing 'chunk' under 'data'

        Returns:
            AssistantEvent with the chunk content, or None if invalid
        """
        chunk = event_data.get("data", {}).get("chunk")
        if chunk and hasattr(chunk, "content") and chunk.content:
            return AssistantEvent(content=chunk.content)
        return None

    def _map_tool_start(self, event_data: dict[str, Any]) -> ToolCallEvent | None:
        """Map on_tool_start event to ToolCallEvent.

        Args:
            event_data: The event data dict containing 'input' under 'data'

        Returns:
            ToolCallEvent with tool info and arguments
        """
        tool_name = event_data.get("name", "")
        if not tool_name:
            return None

        tool_args = event_data.get("data", {}).get("input", {})
        tool_call_id = event_data.get("run_id", "")

        tool_class = self._get_tool_class(tool_name)

        if tool_class:
            args_model = tool_class._get_tool_args_results()[0]
            try:
                args = args_model.model_validate(tool_args)
            except Exception as e:
                logger.warning(
                    "Failed to validate args for tool %s: %s. Falling back to generic args.",
                    tool_name,
                    e,
                )
                args = GenericArgs(args=tool_args)
        else:
            # Fallback for unknown tools
            args = GenericArgs(args=tool_args)
            tool_class = None

        return ToolCallEvent(
            tool_name=tool_name,
            tool_class=tool_class,
            args=args,
            tool_call_id=tool_call_id,
        )

    def _map_tool_end(self, event_data: dict[str, Any]) -> ToolResultEvent | None:
        """Map on_tool_end event to ToolResultEvent.

        Args:
            event_data: The event data dict containing 'output' under 'data'

        Returns:
            ToolResultEvent with tool result
        """
        tool_name = event_data.get("name", "")
        if not tool_name:
            return None

        tool_result = event_data.get("data", {}).get("output", "")
        tool_call_id = event_data.get("run_id", "")

        tool_class = self._get_tool_class(tool_name)

        if tool_class:
            # Get result model from the second element of _get_tool_args_results
            _, result_model = tool_class._get_tool_args_results()
            try:
                result = result_model.model_validate({"output": tool_result})
            except Exception as e:
                logger.warning(
                    "Failed to validate result for tool %s: %s. Falling back to generic result.",
                    tool_name,
                    e,
                )
                result = GenericResult(output=str(tool_result))
        else:
            # Fallback for unknown tools
            result = GenericResult(output=str(tool_result))

        return ToolResultEvent(
            tool_name=tool_name,
            tool_class=tool_class,
            result=result,
            tool_call_id=tool_call_id,
        )

    def _get_tool_class(self, tool_name: str) -> type[BaseTool] | None:
        """Get tool class for a tool name.

        Args:
            tool_name: The name of the tool to look up

        Returns:
            The tool class, or None if not found.
        """
        try:
            tool_instance = self.tool_manager.get(tool_name)
            return tool_instance.__class__
        except Exception as e:
            logger.debug("Could not get tool class for '%s': %s", tool_name, e)
            return None

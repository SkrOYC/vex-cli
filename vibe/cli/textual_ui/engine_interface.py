"""Engine interface protocol for TUI compatibility."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from vibe.core.types import AgentStatsProtocol, BaseEvent


@runtime_checkable
class EngineInterface(Protocol):
    """Minimal interface for VibeLangChainEngine.

    This protocol defines the interface used by the TUI to interact
    with VibeLangChainEngine. The legacy Agent class has been removed
    as part of migration to LangChain 1.2.0.

    Note: The `act()` method was part of the legacy Agent interface
    and is not used by VibeLangChainEngine.
    """

    @property
    def stats(self) -> AgentStatsProtocol:
        """Get current agent statistics."""
        ...

    def run(self, message: str) -> AsyncIterator[BaseEvent]:
        """Run the engine with a message (VibeEngine interface)."""
        ...

    def act(self, message: str) -> AsyncIterator[BaseEvent]:
        """Run the engine with a message (Agent interface)."""
        ...

    async def clear_history(self) -> None:
        """Clear conversation history."""
        ...

    async def compact(self) -> str:
        """Compact the conversation history and return a summary."""
        ...

    def get_log_path(self) -> str | None:
        """Get the path to the interaction log file, if any."""
        ...

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        ...

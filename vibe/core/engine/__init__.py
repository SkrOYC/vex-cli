"""Engine abstraction for Mistral Vibe - LangChain 1.2.0 powered."""

from __future__ import annotations

from vibe.core.engine.langchain_engine import VibeLangChainEngine
from vibe.core.engine.tools import VibeToolAdapter
from vibe.core.engine.tui_events import TUIEventMapper

__all__ = [
    "TUIEventMapper",  # For mapping native LangGraph events
    "VibeEngine",  # Alias for VibeLangChainEngine
    "VibeLangChainEngine",
    "VibeToolAdapter",
]

# Keep VibeEngine as an alias for backward compatibility
VibeEngine = VibeLangChainEngine

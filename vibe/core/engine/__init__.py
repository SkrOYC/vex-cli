"""Engine abstraction for Mistral Vibe - supports both DeepAgents and LangChain 1.2.0."""

from __future__ import annotations

import os

# Feature flag for LangChain 1.2.0 engine
USE_LANGCHAIN = os.getenv("VIBE_USE_LANGCHAIN", "false").lower() == "true"

if USE_LANGCHAIN:
    from vibe.core.engine.langchain_engine import VibeLangChainEngine as VibeEngine
else:
    from vibe.core.engine.engine import VibeEngine  # Legacy DeepAgents engine

from vibe.core.engine.adapters import (
    ApprovalBridge,  # EventTranslator deprecated, kept for backward compat
)
from vibe.core.engine.tools import VibeToolAdapter
from vibe.core.engine.tui_events import TUIEventMapper

__all__ = [
    "ApprovalBridge",  # Deprecated: Use native HumanInTheLoopMiddleware
    "TUIEventMapper",  # New: For mapping native LangGraph events
    "VibeEngine",
    "VibeToolAdapter",
]

"""Engine abstraction for Mistral Vibe - supports both DeepAgents and LangChain 1.2.0."""

from __future__ import annotations

import os

# Feature flag for LangChain 1.2.0 engine
USE_LANGCHAIN = os.getenv("VIBE_USE_LANGCHAIN", "false").lower() == "true"

if USE_LANGCHAIN:
    from vibe.core.engine.langchain_engine import VibeLangChainEngine as VibeEngine
else:
    from vibe.core.engine.engine import VibeEngine  # Legacy DeepAgents engine

from .adapters import ApprovalBridge, EventTranslator
from .tools import VibeToolAdapter

__all__ = ["ApprovalBridge", "EventTranslator", "VibeEngine", "VibeToolAdapter"]
"""State schema for LangChain 1.2.0 Vibe agent.

This module defines the VibeAgentState TypedDict extending the base AgentState
with Vibe-specific fields for token tracking and warnings.
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import NotRequired

from langchain.agents.middleware.types import AgentState as BaseAgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langgraph.channels import EphemeralValue


class VibeAgentState(BaseAgentState):
    """Custom state schema for Vibe agent extending base AgentState.

    Fields:
        warning: User-facing warnings (ephemeral - resets each turn)
        context_tokens: Fallback token count (internal, when usage_metadata unavailable)

    State Evolution:
        - `warning`: Reset to None after each graph step (ephemeral)
        - `context_tokens`: Cumulative across session (fallback only)
    """

    # Ephemeral warning (auto-resets each graph step)
    warning: Annotated[
        str | None,
        EphemeralValue,  # Auto-reset
        PrivateStateAttr,  # Internal only
    ]

    # Fallback token count (only when usage_metadata unavailable)
    context_tokens: Annotated[
        int,
        NotRequired,  # Backward compatible
        PrivateStateAttr,  # Internal middleware field
    ]

"""State schema for LangChain 1.2.0 Vibe agent.

This module defines the VibeAgentState TypedDict extending the base AgentState
with Vibe-specific fields for token tracking and warnings.
"""

from __future__ import annotations

from langchain.agents.middleware.types import AgentState as BaseAgentState


class VibeAgentState(BaseAgentState):
    """Custom state schema for the Vibe agent extending base AgentState.

    This TypedDict adds Vibe-specific fields to the base AgentState schema:
    - context_tokens: Tracks actual token count from usage_metadata
    - warning: Holds warning messages from ContextWarningMiddleware

    The messages field inherits the add_messages reducer from base AgentState,
    which automatically handles appending new messages to the conversation history.

    Usage:
        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[...],
            state_schema=VibeAgentState,
        )
    """

    # Token count from actual usage_metadata (no estimation!)
    # This field is updated by ContextWarningMiddleware and other components
    # to track cumulative token usage across the session
    context_tokens: int

    # Warning messages from ContextWarningMiddleware
    # When context usage reaches threshold, this field holds the warning message
    # that should be displayed to the user
    warning: str | None

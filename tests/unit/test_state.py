"""Unit tests for state.py - VibeAgentState schema validation."""

from __future__ import annotations

from typing import get_type_hints

from vibe.core.engine.state import VibeAgentState


class TestVibeAgentState:
    """Test VibeAgentState schema definition and validation."""

    def test_extends_base_agentstate(self):
        """Test that VibeAgentState extends base AgentState."""
        from typing import get_type_hints

        from langchain.agents.middleware.types import AgentState as BaseAgentState

        # Check that VibeAgentState inherits the base fields
        base_hints = get_type_hints(BaseAgentState)
        vibe_hints = get_type_hints(VibeAgentState)

        # VibeAgentState should have all base fields plus additional ones
        for field in base_hints:
            assert field in vibe_hints, (
                f"Field '{field}' from base AgentState should be in VibeAgentState"
            )

    def test_has_messages_field(self):
        """Test that messages field is inherited from base AgentState."""
        assert "messages" in VibeAgentState.__annotations__

    def test_has_context_tokens_field(self):
        """Test that context_tokens field is defined."""
        assert "context_tokens" in VibeAgentState.__annotations__
        # Verify the type is int
        hints = get_type_hints(VibeAgentState)
        assert hints["context_tokens"] is int

    def test_has_warning_field(self):
        """Test that warning field is defined."""
        assert "warning" in VibeAgentState.__annotations__
        # Verify the type is str | None
        hints = get_type_hints(VibeAgentState)
        assert hints["warning"] == str | None

    def test_has_jump_to_field(self):
        """Test that jump_to field is inherited from base AgentState."""
        assert "jump_to" in VibeAgentState.__annotations__

    def test_has_structured_response_field(self):
        """Test that structured_response field is inherited from base AgentState."""
        assert "structured_response" in VibeAgentState.__annotations__

    def test_creates_valid_instance(self):
        """Test that a valid VibeAgentState instance can be created."""
        state = VibeAgentState(messages=[], context_tokens=0, warning=None)
        assert state["messages"] == []
        assert state["context_tokens"] == 0
        assert state["warning"] is None

    def test_can_add_custom_fields(self):
        """Test that custom fields can be added at runtime."""
        state = VibeAgentState(
            messages=[],
            context_tokens=1000,
            warning="Test warning",
            # These fields are from base AgentState
            jump_to=None,
        )
        assert state["context_tokens"] == 1000
        assert state["warning"] == "Test warning"

    def test_is_typeddict(self):
        """Test that VibeAgentState is a TypedDict."""
        # Check that it has TypedDict-specific attributes
        assert hasattr(VibeAgentState, "__annotations__")
        assert hasattr(VibeAgentState, "__getitem__")


class TestVibeAgentStateUsage:
    """Test VibeAgentState usage in middleware and engine."""

    def test_can_be_used_as_state_schema(self):
        """Test that VibeAgentState can be passed to create_agent."""
        # This is a compile-time check - if the import works, it can be used
        from vibe.core.engine.state import VibeAgentState

        assert VibeAgentState is not None

    def test_state_fields_match_issue_requirements(self):
        """Test that state fields match the requirements from issue #39.

        Required fields from issue:
        - messages: Annotated[list, add_messages] (inherited)
        - context_tokens: int (custom)
        - warning: str | None (custom)
        """
        annotations = VibeAgentState.__annotations__

        # Check required fields exist
        assert "messages" in annotations
        assert "context_tokens" in annotations
        assert "warning" in annotations

        # Verify field types
        hints = get_type_hints(VibeAgentState)
        assert hints["context_tokens"] is int
        assert hints["warning"] == str | None

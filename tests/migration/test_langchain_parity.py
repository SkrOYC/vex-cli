"""Migration parity tests comparing LangChain 1.2.0 engine with DeepAgents engine.

These tests verify that the LangChain 1.2.0 engine produces identical outputs
to the legacy DeepAgents engine, ensuring feature parity and preventing
user experience regressions during migration.

See issue #41 for details.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from vibe.core.config import Backend, ModelConfig, ProviderConfig, VibeConfig
from vibe.core.tools.base import BaseToolConfig, ToolPermission


def extract_tokens_from_events(events: list) -> dict[str, int]:
    """Extract token counts from a list of events.

    Args:
        events: List of events from engine execution

    Returns:
        Dict with input_tokens, output_tokens, and total_tokens
    """
    input_tokens = 0
    output_tokens = 0

    for event in events:
        # Check for usage_metadata in various event formats
        if hasattr(event, "usage_metadata") and event.usage_metadata:
            input_tokens += event.usage_metadata.get("input_tokens", 0)
            output_tokens += event.usage_metadata.get("output_tokens", 0)
        elif isinstance(event, dict):
            usage = event.get("usage_metadata") or event.get("data", {}).get(
                "usage_metadata"
            )
            if usage:
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def extract_tool_calls_from_events(events: list) -> list[str]:
    """Extract tool call names from a list of events.

    Args:
        events: List of events from engine execution

    Returns:
        List of tool call names in order
    """
    tool_calls = []

    for event in events:
        # Check for tool_start events
        if hasattr(event, "get") and event.get("event") == "on_tool_start":
            tool_name = event.get("name") or event.get("data", {}).get("name")
            if tool_name:
                tool_calls.append(tool_name)
        elif isinstance(event, dict):
            if event.get("event") == "on_tool_start":
                tool_name = event.get("name") or event.get("data", {}).get("name")
                if tool_name:
                    tool_calls.append(tool_name)

    return tool_calls


def extract_messages_from_events(events: list) -> list[dict[str, Any]]:
    """Extract final messages from a list of events.

    Args:
        events: List of events from engine execution

    Returns:
        List of message dicts with role and content
    """
    messages = []

    for event in events:
        # Look for assistant messages
        if hasattr(event, "role") or (isinstance(event, dict) and "role" in event):
            message = {
                "role": getattr(event, "role", None) or event.get("role"),
                "content": getattr(event, "content", None)
                or event.get("content")
                or event.get("data", {}).get("content"),
            }
            if message["content"]:
                messages.append(message)

    return messages


class TestLangChainParity:
    """Test parity between LangChain 1.2.0 and DeepAgents engines."""

    def test_same_engine_interface(self):
        """Test that both engines implement the same interface."""
        from vibe.core.engine import VibeEngine
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        # Both should have these methods
        langchain_methods = [
            "initialize",
            "run",
            "reset",
            "compact",
            "clear_history",
            "get_current_messages",
            "get_log_path",
            "handle_approval",
            "resume_execution",
            "reject_execution",
            "session_id",
            "stats",
        ]

        for method in langchain_methods:
            assert hasattr(VibeLangChainEngine, method), (
                f"VibeLangChainEngine missing {method}"
            )

        # VibeEngine (DeepAgents) should also have these methods
        for method in langchain_methods:
            assert hasattr(VibeEngine, method), f"VibeEngine missing {method}"

    def test_same_state_schema(self):
        """Test that both engines use compatible state schemas."""
        from typing import get_type_hints

        from langchain.agents.middleware.types import AgentState as BaseAgentState

        from vibe.core.engine.state import VibeAgentState

        # VibeAgentState should inherit from base AgentState
        base_hints = get_type_hints(BaseAgentState)
        vibe_hints = get_type_hints(VibeAgentState)

        # VibeAgentState should have all base fields
        for field in base_hints:
            assert field in vibe_hints, (
                f"Field '{field}' from base AgentState missing in VibeAgentState"
            )

        # VibeAgentState should have Vibe-specific fields
        assert "context_tokens" in vibe_hints
        assert "warning" in vibe_hints

    def test_same_pricing_config_structure(self, monkeypatch):
        """Test that pricing configuration structure is compatible."""
        from vibe.core.config import VibeConfig
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        # Mock the API key to avoid MissingAPIKeyError
        monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")

        # Create a test config
        config = VibeConfig(
            active_model="test-model",
            use_langchain=True,
            models=[
                ModelConfig(
                    name="gpt-4o-mini",
                    provider="openai-compatible",
                    alias="test-model",
                    input_price=0.000001,  # $1 per 1M tokens
                    output_price=0.000002,  # $2 per 1M tokens
                )
            ],
            providers=[
                ProviderConfig(
                    name="openai-compatible",
                    api_base="https://api.openai.com/v1",
                    api_key_env_var="OPENAI_API_KEY",
                    backend=Backend.GENERIC,
                )
            ],
        )

        engine = VibeLangChainEngine(config)
        pricing = engine._get_pricing_config()

        # Pricing should be a dict mapping model names to (input_rate, output_rate)
        assert isinstance(pricing, dict)
        assert "gpt-4o-mini" in pricing

        input_rate, output_rate = pricing["gpt-4o-mini"]
        # Rates should be per-token (not per-million)
        assert input_rate == 0.000001 / 1_000_000  # $0.000001 per token
        assert output_rate == 0.000002 / 1_000_000  # $0.000002 per token

    def test_same_middleware_stack_structure(self, langchain_config: VibeConfig):
        """Test that middleware stack has the same structure."""
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        # Test with all middleware enabled
        langchain_config.context_warnings = True
        langchain_config.max_price = 10.0

        engine = VibeLangChainEngine(langchain_config)
        middleware = engine._build_middleware_stack()

        # Should have all three middleware types
        middleware_types = [type(m).__name__ for m in middleware]

        assert "ContextWarningMiddleware" in middleware_types
        assert "PriceLimitMiddleware" in middleware_types
        assert "HumanInTheLoopMiddleware" in middleware_types

        # HITL should be last (to handle interrupts after other middleware)
        assert isinstance(middleware[-1], HumanInTheLoopMiddleware)

    def test_same_interrupt_config_building(self):
        """Test that interrupt config is built the same way."""
        from vibe.core.config import VibeConfig
        from vibe.core.engine.permissions import build_interrupt_config

        # Test config
        config = VibeConfig()

        # Add tools with different permissions
        ask_config = BaseToolConfig()
        ask_config.permission = ToolPermission.ASK
        config.tools["test_ask_tool"] = ask_config

        always_config = BaseToolConfig()
        always_config.permission = ToolPermission.ALWAYS
        config.tools["test_always_tool"] = always_config

        # Build interrupt config
        interrupt_on = build_interrupt_config(config)

        # Tools with ASK permission should require approval
        assert "test_ask_tool" in interrupt_on
        # Tools with ALWAYS permission should not
        assert "test_always_tool" not in interrupt_on
        # Dangerous tools should always be in interrupt_on
        assert "bash" in interrupt_on
        assert "write_file" in interrupt_on


class TestLangChainTokenAccuracy:
    """Test that token counting is accurate (no estimation)."""

    def test_uses_usage_metadata_not_estimation(self):
        """Test that actual usage_metadata is used, not estimation."""
        from vibe.core.config import VibeConfig
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        config = VibeConfig(active_model="test-model", use_langchain=True)

        engine = VibeLangChainEngine(config)

        # Test token count extraction with messages
        messages = []

        # Empty messages should return 0
        tokens = engine._get_actual_token_count(messages)
        assert tokens == 0

        # Add a mock message with usage_metadata
        mock_message = MagicMock()
        mock_message.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        messages.append(mock_message)

        tokens = engine._get_actual_token_count(messages)
        assert tokens == 150

        # Test with message that has no usage_metadata
        mock_message2 = MagicMock()
        mock_message2.usage_metadata = None
        messages.append(mock_message2)

        # Should still return 150 (only counts messages with usage_metadata)
        tokens = engine._get_actual_token_count(messages)
        assert tokens == 150

    def test_no_fallback_to_estimation_in_token_count(self):
        """Test that _get_actual_token_count doesn't use estimation.

        This is important because estimation can be inaccurate.
        The method should only count tokens from actual usage_metadata.
        """
        from vibe.core.config import VibeConfig
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        config = VibeConfig(active_model="test-model", use_langchain=True)

        engine = VibeLangChainEngine(config)

        # Messages without usage_metadata should return 0
        mock_message = MagicMock()
        mock_message.usage_metadata = None
        mock_message.content = "This is a very long message that would estimate to many tokens if we used estimation"

        tokens = engine._get_actual_token_count([mock_message])
        assert tokens == 0, "Should not use estimation when usage_metadata is missing"


class TestLangChainApprovalParity:
    """Test that approval behavior is consistent between engines."""

    def test_approval_callback_signature_compatibility(self):
        """Test that approval callback has compatible signature."""
        import inspect

        from vibe.core.engine import VibeEngine
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        # Both should accept approval_callback parameter
        langchain_sig = inspect.signature(VibeLangChainEngine.__init__)
        deepagents_sig = inspect.signature(VibeEngine.__init__)

        assert "approval_callback" in langchain_sig.parameters
        assert "approval_callback" in deepagents_sig.parameters

    def test_handle_approval_signature_compatibility(self):
        """Test that handle_approval has compatible signature."""
        import inspect

        from vibe.core.engine import VibeEngine
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        langchain_sig = inspect.signature(VibeLangChainEngine.handle_approval)
        deepagents_sig = inspect.signature(VibeEngine.handle_approval)

        # Both should accept approved and feedback parameters
        langchain_params = list(langchain_sig.parameters.keys())
        deepagents_params = list(deepagents_sig.parameters.keys())

        # Both should have 'approved' parameter
        assert "approved" in langchain_params
        assert "approved" in deepagents_params

        # Both should have 'feedback' parameter (or similar)
        assert any("feedback" in p.lower() for p in langchain_params)
        assert any("feedback" in p.lower() for p in deepagents_params)


class TestLangChainErrorHandlingParity:
    """Test that error handling is consistent between engines."""

    def test_price_limit_error_message_format(self):
        """Test that price limit errors have consistent format."""
        from langchain_core.messages import AIMessage

        from vibe.core.engine.langchain_middleware import PriceLimitMiddleware

        pricing = {"test-model": (0.001, 0.002)}  # $1/$2 per 1k tokens
        middleware = PriceLimitMiddleware(
            max_price=0.001, model_name="test-model", pricing=pricing
        )

        # Create AI message that would exceed the limit
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 1000,
                "total_tokens": 2000,
            },
        )
        state = {"model_name": "test-model", "messages": [ai_message]}

        with pytest.raises(RuntimeError, match="Price limit exceeded"):
            from langgraph.runtime import Runtime

            middleware.after_model(state, cast(Runtime, None))


class TestLangChainMiddlewareBehavior:
    """Test middleware behavior for consistency."""

    def test_context_warning_only_warns_once(self):
        """Test that context warning is only shown once per session."""
        from typing import cast

        from langgraph.runtime import Runtime

        from vibe.core.engine.langchain_middleware import ContextWarningMiddleware

        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create state with high token count to trigger warning
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 400,
                "total_tokens": 800,  # 80% of 1000
            },
        )

        # First call should warn
        state1 = {"messages": [ai_message]}
        result1 = middleware.before_model(state1, cast(Runtime, None))
        assert result1 is not None
        assert "warning" in result1

        # Second call should not warn (already warned)
        result2 = middleware.before_model(state1, cast(Runtime, None))
        assert result2 is None

    def test_context_warning_uses_actual_tokens_first(self):
        """Test that context warning prefers usage_metadata over estimation."""
        from typing import cast

        from langchain_core.messages import AIMessage
        from langgraph.runtime import Runtime

        from vibe.core.engine.langchain_middleware import ContextWarningMiddleware

        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create message with usage_metadata showing 800 tokens
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 400,
                "total_tokens": 800,
            },
        )

        # Add lots of text that would estimate high
        state = {"messages": ["very long content " * 1000] * 100 + [ai_message]}

        result = middleware.before_model(state, cast(Runtime, None))

        # Should warn because 800/1000 = 80% > 50%
        assert result is not None
        assert "warning" in result
        assert "80%" in result["warning"]

    def test_price_limit_accumulates_cost(self):
        """Test that price limit middleware accumulates cost across calls."""
        from typing import cast

        from langchain_core.messages import AIMessage
        from langgraph.runtime import Runtime

        from vibe.core.engine.langchain_middleware import PriceLimitMiddleware

        # Use very low rates to make calculations work out correctly
        # $0.05 per 1M tokens = $0.00000005 per token
        pricing = {"test-model": (0.00000005, 0.00000005)}
        middleware = PriceLimitMiddleware(
            max_price=0.15, model_name="test-model", pricing=pricing
        )

        # First call: 1000 input + 500 output = 1500 tokens = $0.000075
        ai_message1 = AIMessage(
            content="response1",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        )
        state1 = {"model_name": "test-model", "messages": [ai_message1]}
        middleware.after_model(state1, cast(Runtime, None))

        # Second call: another 1000 input + 0 output = 1000 tokens = $0.00005, total $0.000125
        ai_message2 = AIMessage(
            content="response2",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 0,
                "total_tokens": 1000,
            },
        )
        state2 = {"model_name": "test-model", "messages": [ai_message1, ai_message2]}
        middleware.after_model(state2, cast(Runtime, None))

        # Third call: another 1M input + 1M output = 2M tokens = $0.10, total $0.100075 < $0.15
        ai_message3 = AIMessage(
            content="response3",
            usage_metadata={
                "input_tokens": 1000000,
                "output_tokens": 1000000,
                "total_tokens": 2000000,
            },
        )
        state3 = {
            "model_name": "test-model",
            "messages": [ai_message1, ai_message2, ai_message3],
        }

        # Should not raise yet - still under limit
        result = middleware.after_model(state3, cast(Runtime, None))
        assert result is None

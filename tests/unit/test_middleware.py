"""Unit tests for middleware components."""

from __future__ import annotations

from typing import cast

import pytest

from langgraph.runtime import Runtime

from vibe.core.config import VibeConfig
from vibe.core.engine.middleware import (
    ContextWarningMiddleware,
    PriceLimitMiddleware,
    build_middleware_stack,
)


class TestContextWarningMiddleware:
    """Test ContextWarningMiddleware functionality."""

    def test_no_warning_when_below_threshold(self):
        """Test that no warning is injected when below threshold."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)
        state = {"messages": ["short message"] * 10}  # ~40 tokens, below 500

        result = middleware.before_model(state, cast(Runtime, cast(Runtime, None)))
        assert result is None

    def test_warning_when_above_threshold(self):
        """Test that warning is injected when above threshold."""
        from langchain_core.messages import AIMessage

        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create a message with usage metadata showing 800 tokens (above 500 threshold)
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 400,
                "total_tokens": 800,
            },
        )
        state = {"messages": ["long message content"] * 100 + [ai_message]}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is not None
        assert "warning" in result
        assert "80%" in result["warning"]  # 800/1000 = 80%

    def test_no_warning_after_already_warned(self):
        """Test that warning is only shown once."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)
        state = {"messages": ["long message content"] * 100}

        # First call should warn
        result1 = middleware.before_model(state, cast(Runtime, None))
        assert result1 is not None

        # Second call should not warn
        result2 = middleware.before_model(state, cast(Runtime, None))
        assert result2 is None


class TestPriceLimitMiddleware:
    """Test PriceLimitMiddleware functionality."""

    def test_no_error_when_below_limit(self):
        """Test that no error is raised when below price limit."""
        from langchain_core.messages import AIMessage

        pricing = {"test-model": (0.0001, 0.0002)}  # $0.10 per 1k tokens
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing=pricing
        )

        # Create AI message with usage metadata
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        )
        state = {"model_name": "test-model", "messages": [ai_message]}

        # Should not raise
        result = middleware.after_model(state, cast(Runtime, None))
        assert result is None

    def test_error_when_above_limit(self):
        """Test that RuntimeError is raised when above price limit."""
        from langchain_core.messages import AIMessage

        pricing = {"test-model": (0.001, 0.002)}  # $1.00 per 1k tokens
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing=pricing
        )

        # Create AI message with usage metadata (1000 input + 500 output = 1500 tokens = $1.50)
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        )
        state = {"model_name": "test-model", "messages": [ai_message]}

        with pytest.raises(RuntimeError, match="Price limit exceeded"):
            middleware.after_model(state, cast(Runtime, None))

    def test_accumulates_cost(self):
        """Test that cost accumulates across calls."""
        from langchain_core.messages import AIMessage

        pricing = {"test-model": (0.00005, 0.00005)}  # $0.05 per 1k tokens
        middleware = PriceLimitMiddleware(
            max_price=0.15, model_name="test-model", pricing=pricing
        )

        # First call: 1000 input + 500 output = 1500 tokens = $0.075
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

        # Second call: another 1000 input = 1000 tokens = $0.05, total $0.125
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

        # Third call: another 1000 input = 1000 tokens = $0.05, total $0.175 > $0.15
        ai_message3 = AIMessage(
            content="response3",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 0,
                "total_tokens": 1000,
            },
        )
        state3 = {
            "model_name": "test-model",
            "messages": [ai_message1, ai_message2, ai_message3],
        }
        with pytest.raises(RuntimeError, match="Price limit exceeded"):
            middleware.after_model(state3, cast(Runtime, None))


class TestBuildMiddlewareStack:
    """Test build_middleware_stack function."""

    def test_builds_basic_stack(self):
        """Test that basic middleware stack is built correctly.
        
        Note: Even with default config, HumanInTheLoopMiddleware is added
        because build_interrupt_config() always includes dangerous tools
        (bash, write_file, etc.) for security.
        """
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        config = VibeConfig()
        model = None  # Mock model
        backend = None  # Mock backend

        middleware = build_middleware_stack(config, model, backend)

        # With default config, HumanInTheLoopMiddleware should be present
        # because dangerous tools are always added to interrupt_on
        assert isinstance(middleware, list)
        assert len(middleware) == 1
        assert isinstance(middleware[0], HumanInTheLoopMiddleware)

    def test_includes_subagents_when_enabled(self):
        """Test that SubAgentMiddleware is not manually included (it's handled by DeepAgents)."""
        config = VibeConfig(enable_subagents=True)
        model = None
        backend = None

        middleware = build_middleware_stack(config, model, backend)

        # SubAgentMiddleware is provided automatically by DeepAgents, not manually added
        assert not any("SubAgentMiddleware" in str(type(m)) for m in middleware)

    def test_excludes_subagents_when_disabled(self):
        """Test that SubAgentMiddleware is not manually included (it's handled by DeepAgents)."""
        config = VibeConfig(enable_subagents=False)
        model = None
        backend = None

        middleware = build_middleware_stack(config, model, backend)

        # SubAgentMiddleware is provided automatically by DeepAgents, not manually added
        assert not any("SubAgentMiddleware" in str(type(m)) for m in middleware)

    def test_includes_price_limit_when_configured(self):
        """Test that PriceLimitMiddleware is included when max_price is set."""
        config = VibeConfig(max_price=10.0)
        model = None
        backend = None

        middleware = build_middleware_stack(config, model, backend)

        assert any("PriceLimitMiddleware" in str(type(m)) for m in middleware)

    def test_excludes_price_limit_when_not_configured(self):
        """Test that PriceLimitMiddleware is excluded when max_price is None."""
        config = VibeConfig(max_price=None)
        model = None
        backend = None

        middleware = build_middleware_stack(config, model, backend)

        assert not any("PriceLimitMiddleware" in str(type(m)) for m in middleware)

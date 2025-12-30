"""Unit tests for langchain_middleware.py - LangChain 1.2.0 native middleware."""

from __future__ import annotations

from typing import cast

import pytest

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from vibe.core.engine.langchain_middleware import (
    ContextWarningMiddleware,
    PriceLimitMiddleware,
)


class TestContextWarningMiddleware:
    """Test ContextWarningMiddleware functionality."""

    def test_no_warning_when_below_threshold(self):
        """Test that no warning is injected when below threshold."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)
        
        # Create a mock state with messages but low token count
        state = {"messages": ["short message"] * 10}  # ~40 tokens, below 500 threshold

        result = middleware.before_model(state, cast(Runtime, None))
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

        # First call should warn (will fall back to estimation)
        result1 = middleware.before_model(state, cast(Runtime, None))
        assert result1 is not None

        # Second call should not warn (already warned)
        result2 = middleware.before_model(state, cast(Runtime, None))
        assert result2 is None

    def test_no_warning_when_max_context_none(self):
        """Test that no warning is injected when max_context is None."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=None)
        state = {"messages": ["long message content"] * 1000}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is None

    def test_uses_usage_metadata(self):
        """Test that usage_metadata is preferred over estimation."""
        from langchain_core.messages import AIMessage

        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create a message with usage metadata showing 900 tokens
        ai_message = AIMessage(
            content="short",
            usage_metadata={
                "input_tokens": 450,
                "output_tokens": 450,
                "total_tokens": 900,
            },
        )
        # Add lots of content that would estimate high, but usage_metadata should be used
        state = {"messages": ["very long content " * 1000] * 100 + [ai_message]}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is not None
        assert "warning" in result
        assert "90%" in result["warning"]  # 900/1000 = 90%

    def test_falls_back_to_estimation(self):
        """Test that estimation is used when usage_metadata is not available."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create messages without usage_metadata
        # "short message" is 13 chars, * 2000 = 26k chars, /4 = 6500 tokens
        state = {"messages": ["short message"] * 2000}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is not None
        assert "warning" in result
        assert "650%" in result["warning"]  # 6500/1000 = 650%

    def test_warning_message_format(self):
        """Test warning message is properly formatted."""
        from langchain_core.messages import AIMessage

        middleware = ContextWarningMiddleware(threshold_percent=0.75, max_context=10000)

        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 4000,
                "output_tokens": 4000,
                "total_tokens": 8000,
            },
        )
        state = {"messages": [ai_message]}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is not None
        warning = result["warning"]
        assert "80%" in warning  # 8000/10000 = 80%
        assert "8,000" in warning
        assert "10,000" in warning


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
        middleware = PriceLimitMiddleware(max_price=1.0, model_name="test-model", pricing=pricing)

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

    def test_uses_default_pricing_when_model_not_found(self):
        """Test that default pricing is used when model not in pricing dict."""
        from langchain_core.messages import AIMessage

        pricing = {"other-model": (0.001, 0.002)}
        middleware = PriceLimitMiddleware(max_price=1.0, model_name="test-model", pricing=pricing)

        # Create AI message with unknown model
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        )
        state = {"model_name": "unknown-model", "messages": [ai_message]}

        # Should not raise because default pricing (0.0, 0.0) means free
        result = middleware.after_model(state, cast(Runtime, None))
        assert result is None

    def test_no_error_without_usage_metadata(self):
        """Test that no error is raised when usage_metadata is not present."""
        middleware = PriceLimitMiddleware(
            max_price=0.01, model_name="test-model", pricing={}
        )

        # Create AI message without usage_metadata
        ai_message = AIMessage(content="response")
        state = {"model_name": "test-model", "messages": [ai_message]}

        # Should not raise
        result = middleware.after_model(state, cast(Runtime, None))
        assert result is None

    def test_before_model_returns_none(self):
        """Test that before_model always returns None."""
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing={}
        )
        state = {"messages": []}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is None

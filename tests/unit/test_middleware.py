"""Unit tests for middleware components."""

from __future__ import annotations

import pytest

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

        result = middleware.before_model(state, None)
        assert result is None

    def test_warning_when_above_threshold(self):
        """Test that warning is injected when above threshold."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)
        state = {"messages": ["long message content"] * 100}  # ~800 tokens, above 500

        result = middleware.before_model(state, None)
        assert result is not None
        assert "warning" in result
        assert "50%" in result["warning"]

    def test_no_warning_after_already_warned(self):
        """Test that warning is only shown once."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)
        state = {"messages": ["long message content"] * 100}

        # First call should warn
        result1 = middleware.before_model(state, None)
        assert result1 is not None

        # Second call should not warn
        result2 = middleware.before_model(state, None)
        assert result2 is None


class TestPriceLimitMiddleware:
    """Test PriceLimitMiddleware functionality."""

    def test_no_error_when_below_limit(self):
        """Test that no error is raised when below price limit."""
        pricing = {"test-model": (0.0001, 0.0002)}  # $0.10 per 1k tokens
        middleware = PriceLimitMiddleware(max_price=1.0, pricing=pricing)
        state = {
            "model_name": "test-model",
            "last_usage": {"input_tokens": 1000, "output_tokens": 500},
        }

        # Should not raise
        result = middleware.after_model(state, None)
        assert result is None

    def test_error_when_above_limit(self):
        """Test that RuntimeError is raised when above price limit."""
        pricing = {"test-model": (0.001, 0.002)}  # $1.00 per 1k tokens
        middleware = PriceLimitMiddleware(max_price=1.0, pricing=pricing)
        state = {
            "model_name": "test-model",
            "last_usage": {"input_tokens": 1000, "output_tokens": 500},
        }

        with pytest.raises(RuntimeError, match="Price limit exceeded"):
            middleware.after_model(state, None)

    def test_accumulates_cost(self):
        """Test that cost accumulates across calls."""
        pricing = {"test-model": (0.00005, 0.00005)}  # $0.05 per 1k tokens
        middleware = PriceLimitMiddleware(max_price=0.15, pricing=pricing)

        # First call: 1000 input + 500 output = 1500 tokens = $0.075
        state1 = {
            "model_name": "test-model",
            "last_usage": {"input_tokens": 1000, "output_tokens": 500},
        }
        middleware.after_model(state1, None)

        # Second call: another 1000 input = 1000 tokens = $0.05, total $0.125
        state2 = {
            "model_name": "test-model",
            "last_usage": {"input_tokens": 1000, "output_tokens": 0},
        }
        middleware.after_model(state2, None)

        # Third call: another 1000 input = 1000 tokens = $0.05, total $0.175 > $0.15
        state3 = {
            "model_name": "test-model",
            "last_usage": {"input_tokens": 1000, "output_tokens": 0},
        }
        with pytest.raises(RuntimeError, match="Price limit exceeded"):
            middleware.after_model(state3, None)


class TestBuildMiddlewareStack:
    """Test build_middleware_stack function."""

    def test_builds_basic_stack(self):
        """Test that basic middleware stack is built correctly."""
        config = VibeConfig()
        model = None  # Mock model
        backend = None  # Mock backend

        middleware = build_middleware_stack(config, model, backend)

        # Should have TodoListMiddleware, FilesystemMiddleware, SubAgentMiddleware
        # SummarizationMiddleware is skipped when model is None, ContextWarningMiddleware when context_warnings=False
        assert len(middleware) == 3
        assert any("TodoListMiddleware" in str(type(m)) for m in middleware)
        assert any("FilesystemMiddleware" in str(type(m)) for m in middleware)
        assert any("SubAgentMiddleware" in str(type(m)) for m in middleware)

    def test_includes_subagents_when_enabled(self):
        """Test that SubAgentMiddleware is included when enabled."""
        config = VibeConfig(enable_subagents=True)
        model = None
        backend = None

        middleware = build_middleware_stack(config, model, backend)

        assert any("SubAgentMiddleware" in str(type(m)) for m in middleware)

    def test_excludes_subagents_when_disabled(self):
        """Test that SubAgentMiddleware is excluded when disabled."""
        config = VibeConfig(enable_subagents=False)
        model = None
        backend = None

        middleware = build_middleware_stack(config, model, backend)

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

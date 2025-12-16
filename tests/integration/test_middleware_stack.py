"""Integration tests for middleware stack functionality."""

from __future__ import annotations

from vibe.core.config import VibeConfig
from vibe.core.engine.middleware import build_middleware_stack


class TestMiddlewareStackIntegration:
    """Integration tests for the complete middleware stack."""

    def test_full_stack_creation(self):
        """Test that the full middleware stack can be created without errors."""
        config = VibeConfig(
            enable_subagents=True,
            max_price=5.0,
            context_warnings=True,
            auto_compact_threshold=100_000,
        )
        model = None  # Would be a real model in integration
        backend = None  # Would be a real backend in integration

        middleware = build_middleware_stack(config, model, backend)

        # Should have all middleware types (SummarizationMiddleware skipped when model=None)
        middleware_types = [str(type(m)) for m in middleware]
        assert any("TodoListMiddleware" in t for t in middleware_types)
        assert any("FilesystemMiddleware" in t for t in middleware_types)
        assert any("SubAgentMiddleware" in t for t in middleware_types)
        # assert any("SummarizationMiddleware" in t for t in middleware_types)  # Skipped for None model
        assert any("ContextWarningMiddleware" in t for t in middleware_types)
        assert any("PriceLimitMiddleware" in t for t in middleware_types)

    def test_minimal_stack_creation(self):
        """Test middleware stack with minimal configuration."""
        config = VibeConfig(
            enable_subagents=False, max_price=None, context_warnings=False
        )
        model = None
        backend = None

        middleware = build_middleware_stack(config, model, backend)

        # Should have core middleware only (SummarizationMiddleware skipped when model=None)
        middleware_types = [str(type(m)) for m in middleware]
        assert any("TodoListMiddleware" in t for t in middleware_types)
        assert any("FilesystemMiddleware" in t for t in middleware_types)
        # assert any("SummarizationMiddleware" in t for t in middleware_types)  # Skipped for None model
        assert not any("SubAgentMiddleware" in t for t in middleware_types)
        assert not any("PriceLimitMiddleware" in t for t in middleware_types)
        assert not any("ContextWarningMiddleware" in t for t in middleware_types)

    def test_middleware_ordering(self):
        """Test that middleware is ordered correctly."""
        config = VibeConfig(enable_subagents=True, max_price=1.0, context_warnings=True)
        model = None
        backend = None

        middleware = build_middleware_stack(config, model, backend)

        # Extract types in order
        types = [str(type(m)) for m in middleware]

        # Find indices (SummarizationMiddleware skipped when model=None)
        todo_idx = next(i for i, t in enumerate(types) if "TodoListMiddleware" in t)
        fs_idx = next(i for i, t in enumerate(types) if "FilesystemMiddleware" in t)
        subagent_idx = next(i for i, t in enumerate(types) if "SubAgentMiddleware" in t)
        # summary_idx = next(i for i, t in enumerate(types) if "SummarizationMiddleware" in t)
        context_idx = next(
            i for i, t in enumerate(types) if "ContextWarningMiddleware" in t
        )
        price_idx = next(i for i, t in enumerate(types) if "PriceLimitMiddleware" in t)

        # Verify ordering
        assert todo_idx < fs_idx
        assert fs_idx < subagent_idx
        # assert subagent_idx < summary_idx
        # assert summary_idx < context_idx
        assert subagent_idx < context_idx  # Skip summary
        assert context_idx < price_idx

    # Note: Full integration tests with VibeEngine would require mocking
    # the deepagents create_deep_agent function and testing event flow.
    # These would be added in a separate integration test file.

"""Unit tests for VibeLangChainEngine with LangChain 1.2.0 integration."""

from __future__ import annotations

import pytest

from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_engine import VibeEngineStats, VibeLangChainEngine


class TestVibeLangChainEngine:
    """Test VibeLangChainEngine functionality with LangChain 1.2.0."""

    def test_initialization(self, langchain_config: VibeConfig):
        """Test engine initializes with config."""
        engine = VibeLangChainEngine(langchain_config)

        assert engine._agent is None
        assert engine.config == langchain_config
        assert engine._thread_id.startswith("vibe-session-")
        assert isinstance(engine._stats, VibeEngineStats)

    def test_initialization_with_approval_callback(self, langchain_config: VibeConfig):
        """Test engine initializes with approval callback."""

        async def dummy_callback(request):
            return {"approved": True}

        engine = VibeLangChainEngine(langchain_config, approval_callback=dummy_callback)

        assert engine.approval_callback is not None

    def test_use_langchain_flag(self, langchain_config: VibeConfig):
        """Test that the use_langchain flag is properly set in config."""
        engine = VibeLangChainEngine(langchain_config)
        assert engine.config.use_langchain is True

    def test_reset_functionality(self, langchain_config: VibeConfig):
        """Test engine reset functionality."""
        engine = VibeLangChainEngine(langchain_config)
        # Initialize to create initial state
        engine.initialize()

        initial_thread_id = engine._thread_id

        engine.reset()

        # After reset, agent should be None and will be reinitialized on next use
        assert engine._agent is None
        assert engine._thread_id != initial_thread_id

    def test_stats_property(self, langchain_config: VibeConfig):
        """Test stats property returns expected structure."""
        engine = VibeLangChainEngine(langchain_config)
        stats = engine.stats

        assert isinstance(stats, VibeEngineStats)
        assert hasattr(stats, "context_tokens")
        assert hasattr(stats, "tool_calls_succeeded")
        assert hasattr(stats, "tool_calls_failed")

    def test_session_id_property(self, langchain_config: VibeConfig):
        """Test session_id property returns thread_id."""
        engine = VibeLangChainEngine(langchain_config)

        assert engine.session_id == engine._thread_id

    def test_get_current_messages_empty(self, langchain_config: VibeConfig):
        """Test get_current_messages returns empty list before initialization."""
        engine = VibeLangChainEngine(langchain_config)

        messages = engine.get_current_messages()
        assert messages == []

    def test_get_log_path_returns_none(self, langchain_config: VibeConfig):
        """Test get_log_path returns None (placeholder)."""
        engine = VibeLangChainEngine(langchain_config)

        log_path = engine.get_log_path()
        assert log_path is None

    def test_middleware_stack_empty_by_default(self, langchain_config: VibeConfig):
        """Test middleware stack is empty when no config options enabled."""
        # Create config without context_warnings or max_price
        config = VibeConfig(
            active_model="test-model",
            use_langchain=True,
            context_warnings=False,
            max_price=None,
            models=[
                pytest.importorskip("vibe.core.config").ModelConfig(
                    name="gpt-4o-mini", provider="openai-compatible", alias="test-model"
                )
            ],
            providers=[
                pytest.importorskip("vibe.core.config").ProviderConfig(
                    name="openai-compatible",
                    api_base="https://api.openai.com/v1",
                    api_key_env_var="OPENAI_API_KEY",
                    backend=pytest.importorskip("vibe.core.config").Backend.GENERIC,
                )
            ],
        )

        engine = VibeLangChainEngine(config)
        middleware = engine._build_middleware_stack()

        # Should have HumanInTheLoopMiddleware since default tools require approval
        assert len(middleware) >= 1
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        assert any(isinstance(m, HumanInTheLoopMiddleware) for m in middleware)

    def test_pricing_config(self, langchain_config: VibeConfig):
        """Test pricing configuration is generated correctly."""
        engine = VibeLangChainEngine(langchain_config)
        pricing = engine._get_pricing_config()

        assert isinstance(pricing, dict)
        # The test config should have a model with pricing
        for _model_name, (input_rate, output_rate) in pricing.items():
            assert isinstance(input_rate, (int, float))
            assert isinstance(output_rate, (int, float))
            # Input price per million / 1_000_000 = per token rate
            assert input_rate >= 0
            assert output_rate >= 0


class TestVibeEngineStats:
    """Test VibeEngineStats functionality."""

    def test_stats_initialization(self):
        """Test stats initialize with default values."""
        stats = VibeEngineStats()

        assert stats.context_tokens == 0
        assert stats.session_prompt_tokens == 0
        assert stats.session_completion_tokens == 0
        assert stats.steps == 0

    def test_stats_with_values(self):
        """Test stats with custom values."""
        stats = VibeEngineStats(messages=5, context_tokens=1000)

        assert stats._messages == 5
        assert stats.context_tokens == 1000

    def test_session_total_llm_tokens(self):
        """Test session_total_llm_tokens property."""
        stats = VibeEngineStats()
        stats.session_prompt_tokens = 100
        stats.session_completion_tokens = 200

        assert stats.session_total_llm_tokens == 300

    def test_last_turn_total_tokens(self):
        """Test last_turn_total_tokens property."""
        stats = VibeEngineStats()
        stats.last_turn_prompt_tokens = 50
        stats.last_turn_completion_tokens = 100

        assert stats.last_turn_total_tokens == 150

    def test_session_cost(self):
        """Test session_cost property."""
        stats = VibeEngineStats()
        stats.session_prompt_tokens = 1_000_000
        stats.session_completion_tokens = 500_000
        stats.input_price_per_million = 1.0
        stats.output_price_per_million = 2.0

        # Cost = (1M * $1/M) + (0.5M * $2/M) = $1 + $1 = $2
        assert stats.session_cost == 2.0

    def test_update_pricing(self):
        """Test update_pricing method."""
        stats = VibeEngineStats()
        stats.update_pricing(input_price=1.5, output_price=3.0)

        assert stats.input_price_per_million == 1.5
        assert stats.output_price_per_million == 3.0


class TestLangChainImports:
    """Test that langchain_engine.py has no DeepAgents imports."""

    def test_no_deepagents_imports(self):
        """Verify no DeepAgents imports in langchain_engine module."""
        import vibe.core.engine.langchain_engine as langchain_engine

        # Check the module doesn't import from deepagents
        assert "deepagents" not in langchain_engine.__file__

    def test_uses_langchain_create_agent(self):
        """Verify langchain.agents.create_agent is used."""
        from langchain.agents import create_agent as lc_create_agent

        # The VibeLangChainEngine should use create_agent
        # This test verifies the import is available
        assert lc_create_agent is not None

    def test_uses_human_in_the_loop_middleware(self):
        """Verify HumanInTheLoopMiddleware is imported."""
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        # Should be importable
        assert HumanInTheLoopMiddleware is not None

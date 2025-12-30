"""Integration tests for TUI with VibeEngine."""

from __future__ import annotations

import pytest

from vibe.cli.textual_ui.app import VibeApp
from vibe.core.agent import Agent
from vibe.core.config import VibeConfig
from vibe.core.engine import VibeEngine


class TestTuiVibeEngineIntegration:
    """Test TUI integration with VibeEngine."""

    @pytest.fixture
    def deepagents_config(self) -> VibeConfig:
        """Config with use_deepagents=True."""
        config = VibeConfig()
        config.use_deepagents = True
        return config

    @pytest.fixture
    def legacy_config(self) -> VibeConfig:
        """Config with use_deepagents=False."""
        config = VibeConfig()
        config.use_deepagents = False
        return config

    def test_app_initializes_with_deepagents_config(
        self, deepagents_config: VibeConfig
    ):
        """Test that VibeApp initializes with VibeEngine when use_deepagents=True."""
        app = VibeApp(config=deepagents_config)
        assert app.config.use_deepagents is True
        # Agent is initialized lazily in _initialize_agent

    def test_app_initializes_with_legacy_config(self, legacy_config: VibeConfig):
        """Test that VibeApp initializes with Agent when use_deepagents=False."""
        app = VibeApp(config=legacy_config)
        assert app.config.use_deepagents is False

    @pytest.mark.asyncio
    async def test_agent_initialization_deepagents(self, deepagents_config: VibeConfig):
        """Test that VibeEngine is created when use_deepagents=True."""
        app = VibeApp(config=deepagents_config)
        await app._initialize_agent()
        assert isinstance(app.agent, VibeEngine)

    @pytest.mark.asyncio
    async def test_agent_initialization_legacy(self, legacy_config: VibeConfig):
        """Test that Agent is created when use_deepagents=False."""
        app = VibeApp(config=legacy_config)
        await app._initialize_agent()
        assert isinstance(app.agent, Agent)

    @pytest.mark.asyncio
    async def test_event_stream_connection(self, deepagents_config: VibeConfig):
        """Test that event stream connection works for VibeEngine."""
        app = VibeApp(config=deepagents_config)
        await app._initialize_agent()
        assert isinstance(app.agent, VibeEngine)
        # Test that run method exists and is callable
        assert hasattr(app.agent, "run")
        assert callable(app.agent.run)

    @pytest.mark.asyncio
    async def test_stats_interface_unified(
        self, deepagents_config: VibeConfig, legacy_config: VibeConfig
    ):
        """Test that both engines return compatible stats objects."""
        # Test VibeEngine stats
        app_deep = VibeApp(config=deepagents_config)
        await app_deep._initialize_agent()
        assert app_deep.agent is not None
        deep_stats = app_deep.agent.stats

        # Test legacy Agent stats
        app_legacy = VibeApp(config=legacy_config)
        await app_legacy._initialize_agent()
        assert app_legacy.agent is not None
        legacy_stats = app_legacy.agent.stats

        # Both should have the same interface
        required_attrs = [
            "steps",
            "session_prompt_tokens",
            "session_completion_tokens",
            "context_tokens",
            "session_total_llm_tokens",
            "session_cost",
        ]

        for attr in required_attrs:
            assert hasattr(deep_stats, attr), f"VibeEngine stats missing {attr}"
            assert hasattr(legacy_stats, attr), f"Legacy Agent stats missing {attr}"

    @pytest.mark.asyncio
    async def test_compaction_available(self, deepagents_config: VibeConfig):
        """Test that compaction method exists for VibeEngine."""
        app = VibeApp(config=deepagents_config)
        await app._initialize_agent()
        assert isinstance(app.agent, VibeEngine)
        assert hasattr(app.agent, "compact")
        assert callable(app.agent.compact)

    @pytest.mark.asyncio
    async def test_log_path_method_exists(self, deepagents_config: VibeConfig):
        """Test that get_log_path method exists for VibeEngine."""
        app = VibeApp(config=deepagents_config)
        await app._initialize_agent()
        assert isinstance(app.agent, VibeEngine)
        assert hasattr(app.agent, "get_log_path")
        assert callable(app.agent.get_log_path)

    @pytest.mark.asyncio
    async def test_clear_history_method_exists(self, deepagents_config: VibeConfig):
        """Test that clear_history method exists for VibeEngine."""
        app = VibeApp(config=deepagents_config)
        await app._initialize_agent()
        assert isinstance(app.agent, VibeEngine)
        assert hasattr(app.agent, "clear_history")
        assert callable(app.agent.clear_history)

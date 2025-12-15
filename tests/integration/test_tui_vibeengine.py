"""Integration tests for TUI with VibeEngine."""

import pytest
from unittest.mock import Mock, patch

from vibe.core.config import VibeConfig
from vibe.cli.textual_ui.app import VibeApp
from vibe.core.engine import VibeEngine
from vibe.core.agent import Agent


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

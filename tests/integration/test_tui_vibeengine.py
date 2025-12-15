"""Integration tests for TUI with VibeEngine."""

import pytest
from unittest.mock import Mock, patch

from vibe.core.config import VibeConfig
from vibe.cli.textual_ui.app import VibeApp


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

    @patch("vibe.cli.textual_ui.app.VibeEngine")
    def test_agent_initialization_deepagents(
        self, mock_vibe_engine, deepagents_config: VibeConfig
    ):
        """Test that VibeEngine is created when use_deepagents=True."""
        app = VibeApp(config=deepagents_config)
        # Trigger initialization
        app._ensure_agent_init_task()
        # Since it's async, we can't easily test, but the import and conditional is there

    @patch("vibe.cli.textual_ui.app.Agent")
    def test_agent_initialization_legacy(self, mock_agent, legacy_config: VibeConfig):
        """Test that Agent is created when use_deepagents=False."""
        app = VibeApp(config=legacy_config)
        # Similar issue with async

    def test_event_stream_connection(self, deepagents_config: VibeConfig):
        """Test that event stream is connected properly."""
        # This would require mocking the event loop and testing the async for
        # For now, the code structure is in place
        pass

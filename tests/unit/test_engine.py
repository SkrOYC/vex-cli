"""Unit tests for VibeEngine with DeepAgents integration."""

import pytest
from vibe.core.config import VibeConfig
from vibe.core.engine import VibeEngine
from vibe.core.engine.adapters import ApprovalBridge
from vibe.core.engine.engine import VibeEngineStats


class TestVibeEngine:
    """Test VibeEngine functionality with DeepAgents."""

    def test_initialization(self, deepagents_config: VibeConfig):
        """Test engine initializes with config."""
        engine = VibeEngine(deepagents_config)

        assert engine._agent is None
        engine.initialize()
        assert engine._agent is not None
        assert engine.config == deepagents_config

    def test_initialization_with_approval_bridge(self, deepagents_config: VibeConfig):
        """Test engine initializes with approval bridge."""
        approval_bridge = ApprovalBridge()
        engine = VibeEngine(deepagents_config, approval_callback=approval_bridge)

        assert engine.approval_bridge == approval_bridge

    def test_use_deepagents_flag(self, deepagents_config: VibeConfig):
        """Test that the use_deepagents flag is properly set."""
        engine = VibeEngine(deepagents_config)
        assert engine._use_deepagents is True

    def test_reset_functionality(self, deepagents_config: VibeConfig):
        """Test engine reset functionality."""
        engine = VibeEngine(deepagents_config)
        engine.initialize()

        initial_thread_id = engine._thread_id

        engine.reset()

        # After reset, agent should be None and will be reinitialized on next use
        assert engine._agent is None
        assert engine._thread_id != initial_thread_id

    def test_stats_property(self, deepagents_config: VibeConfig):
        """Test stats property returns expected structure."""
        engine = VibeEngine(deepagents_config)
        stats = engine.stats

        assert isinstance(stats, VibeEngineStats)
        assert hasattr(stats, "context_tokens")
        assert hasattr(stats, "tool_calls_succeeded")
        assert hasattr(stats, "tool_calls_failed")

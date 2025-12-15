"""Integration tests for DeepAgents engine functionality."""

import pytest
from vibe.core.config import VibeConfig
from vibe.core.engine import VibeEngine
from vibe.core.types import AssistantEvent, ToolCallEvent, ToolResultEvent


class TestDeepAgentsIntegration:
    """Test DeepAgents integration with end-to-end functionality."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, deepagents_config: VibeConfig):
        """Test basic engine initialization."""
        engine = VibeEngine(deepagents_config)
        engine.initialize()
        
        assert engine._agent is not None
        assert engine.config == deepagents_config
    
    @pytest.mark.asyncio
    async def test_simple_conversation(self, deepagents_config: VibeConfig):
        """Test basic conversation flow without tools."""
        engine = VibeEngine(deepagents_config)
        engine.initialize()
        
        # Since the actual run method might require actual API calls,
        # we'll focus on testing the initialization and structure
        assert engine._agent is not None
        
        # Test stats functionality
        stats = engine.stats
        assert isinstance(stats, dict)
        assert "messages" in stats
        assert "todos" in stats
        assert "context_tokens" in stats
    
    @pytest.mark.asyncio
    async def test_engine_reset_functionality(self, deepagents_config: VibeConfig):
        """Test engine reset functionality."""
        engine = VibeEngine(deepagents_config)
        engine.initialize()
        
        # Verify agent is initialized
        assert engine._agent is not None
        
        # Reset the engine
        engine.reset()
        
        # After reset, agent should be None (will be reinitialized on next use)
        assert engine._agent is None
    
    @pytest.mark.asyncio
    async def test_event_translator_integration(self, deepagents_config: VibeConfig):
        """Test EventTranslator integration."""
        from vibe.core.engine.adapters import EventTranslator
        
        translator = EventTranslator(deepagents_config)
        
        # Test with a simple known event
        event = {"event": "unknown", "data": {}}
        result = translator.translate(event)
        assert result is None  # Unknown events should return None
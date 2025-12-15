"""Unit tests for DeepAgentsConfig with DeepAgents integration."""

import pytest
from vibe.core.config import VibeConfig, BaseToolConfig
from vibe.core.tools.base import ToolPermission
from vibe.core.engine.config_bridge import DeepAgentsConfig


class TestDeepAgentsConfigBridge:
    """Test DeepAgentsConfig bridge functionality."""
    
    def test_create_model(self, deepagents_config: VibeConfig):
        """Test model creation from config."""
        model = DeepAgentsConfig.create_model(deepagents_config)
        # Just verify it returns something - actual model validation happens elsewhere
        assert model is not None
    
    def test_create_backend(self, deepagents_config: VibeConfig):
        """Test backend creation from config."""
        backend = DeepAgentsConfig.create_backend(deepagents_config)
        # Just verify it returns something - actual backend validation happens elsewhere
        assert backend is not None
    
    def test_create_interrupt_config_with_ask_permission(self, deepagents_config: VibeConfig):
        """Test that tools with 'ask' permission are properly mapped to interrupts."""
        config = VibeConfig(
            tools={
                "write_file": BaseToolConfig(permission=ToolPermission.ASK),
                "bash": BaseToolConfig(permission=ToolPermission.ASK),
                "read_file": BaseToolConfig(permission=ToolPermission.ALWAYS),
                "search": BaseToolConfig(permission=ToolPermission.NEVER),
            }
        )
        
        interrupt_config = DeepAgentsConfig.create_interrupt_config(config)
        
        # Only tools with 'ask' permission should be in interrupt_config
        assert "write_file" in interrupt_config
        assert interrupt_config["write_file"] is True
        assert "bash" in interrupt_config
        assert interrupt_config["bash"] is True
        
        # Tools with 'always' permission should not be in interrupt_config
        assert "read_file" not in interrupt_config
        
        # Tools with 'never' permission should not be in interrupt_config
        assert "search" not in interrupt_config
    
    def test_create_interrupt_config_empty_tools(self, deepagents_config: VibeConfig):
        """Test interrupt config creation with empty tools."""
        config = VibeConfig(tools={})
        interrupt_config = DeepAgentsConfig.create_interrupt_config(config)
        
        assert interrupt_config == {}
    
    def test_create_interrupt_config_no_ask_tools(self, deepagents_config: VibeConfig):
        """Test interrupt config when no tools have 'ask' permission."""
        config = VibeConfig(
            tools={
                "read_file": BaseToolConfig(permission=ToolPermission.ALWAYS),
                "list_files": BaseToolConfig(permission=ToolPermission.ALWAYS),
            }
        )
        
        interrupt_config = DeepAgentsConfig.create_interrupt_config(config)
        
        assert interrupt_config == {}
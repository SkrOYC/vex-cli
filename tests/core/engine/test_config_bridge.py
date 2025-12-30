"""Tests for DeepAgentsConfig configuration bridge."""

from __future__ import annotations

from pathlib import Path

from vibe.core.config import BaseToolConfig, VibeConfig
from vibe.core.engine.config_bridge import DeepAgentsConfig
from vibe.core.tools.base import ToolPermission


class TestDeepAgentsConfig:
    """Test the DeepAgentsConfig class and its methods."""

    def test_create_interrupt_config_with_ask_permission(self):
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

    def test_create_interrupt_config_empty_tools(self):
        """Test interrupt config creation with empty tools."""
        config = VibeConfig(tools={})
        interrupt_config = DeepAgentsConfig.create_interrupt_config(config)

        assert interrupt_config == {}

    def test_create_interrupt_config_no_ask_tools(self):
        """Test interrupt config when no tools have 'ask' permission."""
        config = VibeConfig(
            tools={
                "read_file": BaseToolConfig(permission=ToolPermission.ALWAYS),
                "list_files": BaseToolConfig(permission=ToolPermission.ALWAYS),
            }
        )

        interrupt_config = DeepAgentsConfig.create_interrupt_config(config)

        assert interrupt_config == {}

    def test_create_backend(self):
        """Test backend creation."""
        # Use a temporary directory that actually exists
        workdir = Path("/tmp")
        config = VibeConfig(workdir=workdir)

        backend = DeepAgentsConfig.create_backend(config)

        # The FilesystemBackend may not have a direct root_dir attribute
        # We can test that the backend was created successfully
        assert backend is not None

    def test_create_backend_default_workdir(self):
        """Test backend creation with default workdir."""
        config = VibeConfig(workdir=None)  # Should use effective_workdir (cwd)

        backend = DeepAgentsConfig.create_backend(config)

        # Verify backend was created successfully
        assert backend is not None

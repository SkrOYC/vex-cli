"""Unit tests for permission checking and pattern matching."""

import pytest
from vibe.core.config import VibeConfig
from vibe.core.engine.permissions import (
    check_allowlist_denylist,
    matches_pattern,
    build_interrupt_config,
)
from vibe.core.tools.base import BaseToolConfig, ToolPermission


class TestPermissions:
    """Test permission checking functionality."""

    def test_check_allowlist_denylist_no_config(self):
        """Test checking permissions when tool has no config."""
        config = VibeConfig()
        result = check_allowlist_denylist("unknown_tool", {}, config)
        assert result == ToolPermission.ASK

    def test_check_allowlist_denylist_allowlist_match(self):
        """Test allowlist pattern matching."""
        config = VibeConfig()
        tool_config = BaseToolConfig()
        tool_config.permission = ToolPermission.ASK
        tool_config.allowlist = ["*.txt"]
        config.tools["write_file"] = tool_config

        # Should allow .txt files
        result = check_allowlist_denylist("write_file", {"path": "/test.txt"}, config)
        assert result == ToolPermission.ALWAYS

    def test_check_allowlist_denylist_denylist_match(self):
        """Test denylist pattern matching."""
        config = VibeConfig()
        tool_config = BaseToolConfig()
        tool_config.permission = ToolPermission.ASK
        tool_config.denylist = ["*.txt"]
        config.tools["write_file"] = tool_config

        # Should deny .txt files
        result = check_allowlist_denylist("write_file", {"path": "/test.txt"}, config)
        assert result == ToolPermission.NEVER

    def test_check_allowlist_denylist_no_match(self):
        """Test when no patterns match."""
        config = VibeConfig()
        tool_config = BaseToolConfig()
        tool_config.permission = ToolPermission.ASK
        tool_config.denylist = ["*.exe"]
        tool_config.allowlist = ["*.txt"]
        config.tools["write_file"] = tool_config

        # .py file doesn't match either pattern
        result = check_allowlist_denylist("write_file", {"path": "/test.py"}, config)
        assert result == ToolPermission.ASK

    def test_matches_pattern_regex(self):
        """Test regex pattern matching."""
        # Regex pattern (enclosed in slashes)
        assert matches_pattern("write_file", {"path": "/test.txt"}, "/.*\\.txt$/")
        assert not matches_pattern("write_file", {"path": "/test.py"}, "/.*\\.txt$/")

    def test_matches_pattern_glob(self):
        """Test glob pattern matching."""
        assert matches_pattern("write_file", {"path": "/home/user/test.txt"}, "*.txt")
        assert not matches_pattern(
            "write_file", {"path": "/home/user/test.py"}, "*.txt"
        )

    def test_matches_pattern_simple_string(self):
        """Test simple string matching."""
        assert matches_pattern("bash", {"command": "rm -rf /"}, "rm")
        assert not matches_pattern("bash", {"command": "ls -la"}, "rm")

    def test_build_interrupt_config(self):
        """Test building interrupt config from tool permissions."""
        config = VibeConfig()

        write_config = BaseToolConfig()
        write_config.permission = ToolPermission.ASK
        config.tools["write_file"] = write_config

        read_config = BaseToolConfig()
        read_config.permission = ToolPermission.ALWAYS
        config.tools["read_file"] = read_config

        result = build_interrupt_config(config)

        # write_file should be interrupted (ASK)
        assert "write_file" in result
        assert result["write_file"] is True

        # read_file should not be interrupted (ALWAYS)
        assert "read_file" not in result

        # Dangerous tools should be added by default
        assert "bash" in result
        assert "execute" in result

    def test_matches_pattern_invalid_regex(self):
        """Test pattern matching with invalid regex."""
        # Invalid regex should not match
        assert not matches_pattern("tool", {}, "/invalid[regex/")
        assert not matches_pattern("tool", {}, "/(?P<invalid/")

    def test_matches_pattern_empty_patterns(self):
        """Test pattern matching with empty or None patterns."""
        # Empty patterns should not match
        assert not matches_pattern("tool", {"path": "/file.txt"}, "")
        assert not matches_pattern("tool", {"path": "/file.txt"}, None)

    def test_matches_pattern_complex_file_paths(self):
        """Test pattern matching with complex file paths."""
        # Paths with special characters
        assert matches_pattern(
            "write_file", {"path": "/home/user/file with spaces.txt"}, "*.txt"
        )
        assert matches_pattern(
            "write_file", {"path": "/home/user/file[1].txt"}, "*.txt"
        )
        assert matches_pattern(
            "write_file", {"path": "/home/user/file(1).txt"}, "*.txt"
        )

        # Unicode characters
        assert matches_pattern("write_file", {"path": "/home/user/文件.txt"}, "*.txt")

        # Absolute vs relative paths
        assert matches_pattern(
            "write_file", {"path": "relative/path/file.txt"}, "*.txt"
        )
        assert matches_pattern(
            "write_file", {"path": "./relative/path/file.txt"}, "*.txt"
        )

    def test_matches_pattern_regex_edge_cases(self):
        """Test regex pattern matching edge cases."""
        # Regex with special characters
        assert matches_pattern("bash", {"command": "rm -rf /tmp/*"}, "/rm -rf.*/")
        assert not matches_pattern("bash", {"command": "ls -la"}, "/rm -rf.*/")

        # Case insensitive matching - using simple pattern for now
        assert matches_pattern(
            "tool", {"arg": "TestValue"}, "/TestValue/"
        )  # Exact case match

    def test_check_allowlist_denylist_complex_scenarios(self):
        """Test allowlist/denylist with complex scenarios."""
        config = VibeConfig()

        # Tool with both allowlist and denylist
        tool_config = BaseToolConfig()
        tool_config.permission = ToolPermission.ASK
        tool_config.allowlist = ["*.txt", "*.md"]
        tool_config.denylist = ["*secret*"]
        config.tools["write_file"] = tool_config

        # Should deny secret files even if they match allowlist
        result = check_allowlist_denylist(
            "write_file", {"path": "/docs/secret.txt"}, config
        )
        assert result == ToolPermission.NEVER

        # Should allow non-secret txt files
        result = check_allowlist_denylist(
            "write_file", {"path": "/docs/public.txt"}, config
        )
        assert result == ToolPermission.ALWAYS

        # Should deny secret md files
        result = check_allowlist_denylist(
            "write_file", {"path": "/docs/secret.md"}, config
        )
        assert result == ToolPermission.NEVER

        # Should ask for files that don't match any pattern
        result = check_allowlist_denylist(
            "write_file", {"path": "/docs/file.py"}, config
        )
        assert result == ToolPermission.ASK

    def test_build_interrupt_config_edge_cases(self):
        """Test interrupt config building with edge cases."""
        config = VibeConfig()

        # Tool with NEVER permission
        never_config = BaseToolConfig()
        never_config.permission = ToolPermission.NEVER
        config.tools["dangerous_tool"] = never_config

        # Tool with ALWAYS permission
        always_config = BaseToolConfig()
        always_config.permission = ToolPermission.ALWAYS
        config.tools["safe_tool"] = always_config

        result = build_interrupt_config(config)

        # NEVER tools should not be interrupted (not in config)
        assert "dangerous_tool" not in result

        # ALWAYS tools should not be interrupted
        assert "safe_tool" not in result

        # ASK tools should be interrupted
        ask_config = BaseToolConfig()
        ask_config.permission = ToolPermission.ASK
        config.tools["ask_tool"] = ask_config

        result = build_interrupt_config(config)
        assert "ask_tool" in result
        assert result["ask_tool"] is True

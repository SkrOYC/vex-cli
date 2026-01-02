"""Permission checking and pattern matching for tool approvals."""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from vibe.core.config import VibeConfig
from vibe.core.tools.base import ToolPermission


def check_allowlist_denylist(
    tool_name: str, args: dict[str, Any], config: VibeConfig
) -> ToolPermission:
    """Check if tool call matches allowlist/denylist patterns.

    Returns:
        ToolPermission.ALWAYS if allowlisted
        ToolPermission.NEVER if denylisted
        Original permission if no match
    """
    tool_config = config.tools.get(tool_name)
    if not tool_config:
        return ToolPermission.ASK  # Default if no config

    # Check denylist first (higher priority)
    for pattern in tool_config.denylist:
        if matches_pattern(tool_name, args, pattern):
            return ToolPermission.NEVER

    # Check allowlist
    for pattern in tool_config.allowlist:
        if matches_pattern(tool_name, args, pattern):
            return ToolPermission.ALWAYS

    # No match, return original permission
    return tool_config.permission


def matches_pattern(tool_name: str, args: dict[str, Any], pattern: str) -> bool:  # noqa: PLR0911, PLR0912
    """Check if tool call matches a pattern.

    Supports:
    - Regex patterns (if starts/ends with /, supports flags like /pattern/i)
    - Glob patterns for file paths
    - Simple string matching
    """
    # Handle empty or None patterns
    if not pattern or pattern is None:
        return False

    # Regex pattern (enclosed in forward slashes)
    if pattern.startswith("/") and pattern.endswith("/"):
        content = pattern[1:-1]  # Remove enclosing slashes

        # Check if there's a trailing slash indicating flags
        parts = content.rsplit("/", 1)
        REGEX_PATTERN_PARTS_COUNT = 2
        if len(parts) == REGEX_PATTERN_PARTS_COUNT and all(
            c in "imsxau" for c in parts[1]
        ):
            # Has flags: /pattern/flags
            regex_pattern, flag_str = parts
            flags = 0
            for flag in flag_str:
                if flag == "i":
                    flags |= re.IGNORECASE
                # Add other flags as needed
        else:
            regex_pattern = content
            flags = 0

        try:
            compiled_regex = re.compile(regex_pattern, flags)
            # Check tool name
            if compiled_regex.search(
                tool_name
            ):  # Use search instead of match for substring
                return True
            # Check args values
            for arg_value in args.values():
                if isinstance(arg_value, str) and compiled_regex.search(arg_value):
                    return True
        except re.error:
            pass  # Invalid regex, fall through

    # Glob pattern - primarily for file paths
    # Check common file path args
    file_path_args = ["path", "file_path", "filepath", "directory", "dir"]
    for arg_name in file_path_args:
        if arg_name in args and isinstance(args[arg_name], str):
            path_str = args[arg_name]
            # Try matching against full path first
            if fnmatch.fnmatch(path_str, pattern):
                return True
            # Also try matching against basename for file patterns
            import os

            basename = os.path.basename(path_str)
            if fnmatch.fnmatch(basename, pattern):
                return True

    # Simple string matching as fallback
    # Check tool name
    if pattern.lower() in tool_name.lower():
        return True
    # Check args values
    for arg_value in args.values():
        if isinstance(arg_value, str) and pattern.lower() in arg_value.lower():
            return True

    return False


def get_effective_permission(
    tool_name: str, args: dict[str, Any], config: VibeConfig
) -> ToolPermission:
    """Get effective permission for a tool call, considering patterns.

    This function evaluates allowlist/denylist patterns at interrupt time
    to determine if a tool call should be auto-approved, auto-rejected,
    or require user approval.

    Args:
        tool_name: Name of the tool being called
        args: Arguments passed to the tool
        config: VibeConfig containing tool permissions and patterns

    Returns:
        ToolPermission.ALWAYS if allowlisted or base permission is ALWAYS
        ToolPermission.NEVER if denylisted or base permission is NEVER
        ToolPermission.ASK if no pattern matches and base permission is ASK
    """
    # First check if tool has explicit config
    tool_config = config.tools.get(tool_name)
    if not tool_config:
        # Default to ASK for unknown tools
        return ToolPermission.ASK

    # Check pattern-based permissions first (highest priority)
    pattern_permission = check_allowlist_denylist(tool_name, args, config)
    if pattern_permission != tool_config.permission:
        # Pattern override takes precedence
        return pattern_permission

    # Fall back to base permission
    return tool_config.permission


def build_interrupt_config(config: VibeConfig) -> dict[str, Any]:
    """Build interrupt config from Vibe tool permissions with pattern support."""
    interrupt_on = {}

    # First, add default dangerous tools to ensure they always interrupt
    dangerous_tools = ["create", "edit", "edit_file", "bash", "execute"]
    for tool in dangerous_tools:
        interrupt_on[tool] = True

    # Then process explicitly configured tools
    for tool_name, tool_config in config.tools.items():
        # Get effective permission (considering patterns)
        # For interrupt config, we use the base permission since patterns are checked at runtime
        permission = tool_config.permission

        match permission:
            case ToolPermission.ALWAYS:
                # No interrupt needed - remove from dangerous tools list if present
                interrupt_on.pop(tool_name, None)
            case ToolPermission.ASK:
                # Interrupt before execution
                interrupt_on[tool_name] = True
            case ToolPermission.NEVER:
                # Tool filtered out (not added to agent) - remove from interrupt config
                interrupt_on.pop(tool_name, None)

    return interrupt_on

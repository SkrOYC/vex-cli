"""Pytest configuration for filesystem tools tests.

This conftest provides fixtures for filesystem tool testing without requiring
heavy engine dependencies that may not be available in all environments.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from vibe.core.tools.filesystem.grep import (
    GrepArgs,
    GrepResult,
    GrepTool,
    GrepToolConfig,
    GrepToolState,
)
from vibe.core.tools.filesystem.list_files import (
    ListFilesArgs,
    ListFilesResult,
    ListFilesTool,
    ListFilesToolConfig,
    ListFilesToolState,
)


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for file operations."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tool_config(temp_dir: Path) -> ListFilesToolConfig:
    """Create a ListFilesToolConfig."""
    return ListFilesToolConfig(workdir=temp_dir)


@pytest.fixture
def tool_state() -> ListFilesToolState:
    """Create a fresh ListFilesToolState for each test."""
    return ListFilesToolState()


@pytest.fixture
def tool(
    tool_config: ListFilesToolConfig, tool_state: ListFilesToolState
) -> ListFilesTool:
    """Create a ListFilesTool instance for testing."""
    return ListFilesTool(config=tool_config, state=tool_state)


# =============================================================================
# GrepTool Fixtures
# =============================================================================


@pytest.fixture
def grep_config(temp_dir: Path) -> GrepToolConfig:
    """Create a GrepToolConfig for testing."""
    return GrepToolConfig(workdir=temp_dir)


@pytest.fixture
def grep_state() -> GrepToolState:
    """Create a fresh GrepToolState for each test."""
    return GrepToolState()


@pytest.fixture
def grep_tool(grep_config: GrepToolConfig, grep_state: GrepToolState) -> GrepTool:
    """Create a GrepTool instance for testing."""
    return GrepTool(config=grep_config, state=grep_state)


# Re-export commonly used classes for convenience
__all__ = [
    "GrepArgs",
    "GrepResult",
    "GrepTool",
    "GrepToolConfig",
    "GrepToolState",
    "ListFilesArgs",
    "ListFilesResult",
    "ListFilesTool",
    "ListFilesToolConfig",
    "ListFilesToolState",
    "grep_config",
    "grep_state",
    "grep_tool",
    "temp_dir",
    "tool",
    "tool_config",
    "tool_state",
]

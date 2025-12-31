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
)
from vibe.core.tools.filesystem.list_files import (
    ListFilesArgs,
    ListFilesResult,
    ListFilesTool,
)


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for file operations."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tool(temp_dir: Path) -> ListFilesTool:
    """Create a ListFilesTool instance for testing."""
    return ListFilesTool(workdir=temp_dir)


# =============================================================================
# GrepTool Fixtures
# =============================================================================


@pytest.fixture
def grep_tool(temp_dir: Path) -> GrepTool:
    """Create a GrepTool instance for testing.

    Uses the same temp_dir as the test to ensure file visibility.
    """
    return GrepTool(workdir=temp_dir)


# Re-export commonly used classes for convenience
__all__ = [
    "GrepArgs",
    "GrepResult",
    "GrepTool",
    "ListFilesArgs",
    "ListFilesResult",
    "ListFilesTool",
    "grep_tool",
    "temp_dir",
    "tool",
]

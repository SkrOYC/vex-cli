"""Shared fixtures for performance benchmarks."""

import json
import pytest
from pathlib import Path

from vibe.core.config import VibeConfig, SessionLoggingConfig


@pytest.fixture
def config(tmp_path: Path) -> VibeConfig:
    """Create a test configuration for benchmarks."""
    # Use defaults but override workdir and disable logging
    return VibeConfig(
        workdir=tmp_path,
        session_logging=SessionLoggingConfig(enabled=False),
        auto_compact_threshold=0,
        context_warnings=False,
    )


@pytest.fixture
def short_conversation():
    """Load short conversation fixture."""
    return load_fixture("short_conversation.json")


@pytest.fixture
def long_conversation():
    """Load long conversation fixture."""
    return load_fixture("long_conversation.json")


@pytest.fixture
def complex_task():
    """Load complex task fixture."""
    return load_fixture("complex_task.json")


@pytest.fixture
def tool_scenarios():
    """Load tool scenarios fixture."""
    return load_fixture("tool_scenarios.json")


def load_fixture(filename: str) -> dict:
    """Load a JSON fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / filename
    with open(fixture_path, "r") as f:
        return json.load(f)
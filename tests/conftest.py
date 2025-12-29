from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest
import tomli_w

from vibe.core import config_path
from vibe.core.config import VibeConfig, ModelConfig, ProviderConfig, Backend
from vibe.core.engine import VibeEngine
from vibe.core.engine.adapters import EventTranslator


def get_base_config() -> dict[str, Any]:
    return {
        "active_model": "devstral-latest",
        "providers": [
            {
                "name": "mistral",
                "api_base": "https://api.mistral.ai/v1",
                "api_key_env_var": "MISTRAL_API_KEY",
                "backend": "mistral",
            }
        ],
        "models": [
            {
                "name": "mistral-vibe-cli-latest",
                "provider": "mistral",
                "alias": "devstral-latest",
            }
        ],
    }


@pytest.fixture(autouse=True)
def config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    tmp_path = tmp_path_factory.mktemp("vibe")
    config_dir = tmp_path / ".vibe"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(tomli_w.dumps(get_base_config()), encoding="utf-8")

    monkeypatch.setattr(config_path, "_DEFAULT_VIBE_HOME", config_dir)
    return config_dir


@pytest.fixture(autouse=True)
def _mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "mock")


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock platform to be Linux with /bin/sh shell for consistent test behavior.

    This ensures that platform-specific system prompt generation is consistent
    across all tests regardless of the actual platform running the tests.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SHELL", "/bin/sh")


@pytest.fixture
def deepagents_config(monkeypatch: pytest.MonkeyPatch) -> VibeConfig:
    """Create test configuration for DeepAgents."""
    # Mock the API key to avoid MissingAPIKeyError
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    
    return VibeConfig(
        active_model="test-model",  # Must match the alias
        use_deepagents=True,
        models=[
            ModelConfig(
                name="gpt-4o-mini",
                provider="openai-compatible",
                alias="test-model"
            )
        ],
        providers=[
            ProviderConfig(
                name="openai-compatible",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                backend=Backend.GENERIC,
            )
        ]
    )


@pytest.fixture
def langchain_config(monkeypatch: pytest.MonkeyPatch) -> VibeConfig:
    """Create test configuration for LangChain 1.2.0 engine."""
    # Mock the API key to avoid MissingAPIKeyError
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    
    return VibeConfig(
        active_model="test-model",  # Must match the alias
        use_langchain=True,
        models=[
            ModelConfig(
                name="gpt-4o-mini",
                provider="openai-compatible",
                alias="test-model"
            )
        ],
        providers=[
            ProviderConfig(
                name="openai-compatible",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                backend=Backend.GENERIC,
            )
        ]
    )


@pytest.fixture
def vibe_engine(deepagents_config: VibeConfig) -> VibeEngine:
    """Create VibeEngine for testing."""
    return VibeEngine(deepagents_config)


@pytest.fixture
def event_translator(deepagents_config: VibeConfig) -> EventTranslator:
    """Create EventTranslator for testing."""
    return EventTranslator(deepagents_config)

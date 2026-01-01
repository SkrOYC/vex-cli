from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import pytest
import tomli_w

# Try to import heavy dependencies, but fail gracefully if they're not available
# This allows filesystem tests to run without needing all ML dependencies
try:
    from vibe.core import config_path
    from vibe.core.config import Backend, ModelConfig, ProviderConfig, VibeConfig
    from vibe.core.engine import VibeEngine

    _HEAVY_IMPORTS_AVAILABLE = True
except ImportError as e:
    _HEAVY_IMPORTS_AVAILABLE = False
    _IMPORT_ERROR = e

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig
    from vibe.core.engine import VibeEngine


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
        models=[
            ModelConfig(
                name="gpt-4o-mini", provider="openai-compatible", alias="test-model"
            )
        ],
        providers=[
            ProviderConfig(
                name="openai-compatible",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                backend=Backend.GENERIC,
            )
        ],
    )


@pytest.fixture
def langchain_config(monkeypatch: pytest.MonkeyPatch) -> VibeConfig:
    """Create test configuration for LangChain 1.2.0 engine."""
    # Mock the API key to avoid MissingAPIKeyError
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")

    return VibeConfig(
        active_model="test-model",  # Must match the alias
        models=[
            ModelConfig(
                name="gpt-4o-mini", provider="openai-compatible", alias="test-model"
            )
        ],
        providers=[
            ProviderConfig(
                name="openai-compatible",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                backend=Backend.GENERIC,
            )
        ],
    )


@pytest.fixture
def vibe_engine(langchain_config: VibeConfig) -> VibeEngine:
    """Create VibeEngine for testing."""
    return VibeEngine(langchain_config)


@pytest.fixture
def fake_interrupted_agent():
    """Create a fake agent with interrupt support."""
    from tests.stubs.fake_backend import FakeInterruptedAgent

    return FakeInterruptedAgent()


@pytest.fixture
def hitl_engine(langchain_config: VibeConfig):
    """Fixture to create a VibeLangChainEngine with a fake interrupted agent."""
    from tests.stubs.fake_backend import FakeInterruptedAgent
    from vibe.core.engine.langchain_engine import VibeLangChainEngine

    engine = VibeLangChainEngine(langchain_config)
    engine.initialize()
    fake_agent = FakeInterruptedAgent()
    engine._agent = fake_agent  # type: ignore[assignment]
    return engine, fake_agent


@pytest.fixture
def fake_engine_with_events():
    """Create FakeVibeLangChainEngine with configurable synthetic events.

    Usage:
        def test_example(fake_engine_with_events):
            engine = fake_engine_with_events(
                events_to_yield=[AssistantEvent(content="Hello")]
            )
            async for event in engine.run("Test"):
                process(event)
    """
    from tests.stubs.fake_backend import FakeVibeLangChainEngine

    def _create_engine(events_to_yield: list | None = None):
        engine = FakeVibeLangChainEngine(
            config=langchain_config(),
            events_to_yield=events_to_yield,
        )
        engine.initialize()
        return engine

    return _create_engine


@pytest.fixture
def fake_engine_basic_response():
    """Create FakeVibeLangChainEngine with basic assistant response."""
    from tests.stubs.fake_backend import FakeVibeLangChainEngine
    from vibe.core.types import AssistantEvent

    engine = FakeVibeLangChainEngine(
        config=langchain_config(),
        events_to_yield=[
            AssistantEvent(content="This is a test response"),
        ],
    )
    engine.initialize()
    return engine


@pytest.fixture
def fake_engine_with_tool_call():
    """Create FakeVibeLangChainEngine with tool execution flow."""
    from tests.stubs.fake_backend import FakeVibeLangChainEngine
    from vibe.core.types import AssistantEvent, ToolCallEvent, ToolResultEvent
    from vibe.core.tools.filesystem.bash import BashTool

    engine = FakeVibeLangChainEngine(
        config=langchain_config(),
        events_to_yield=[
            AssistantEvent(content="I'll execute a command"),
            ToolCallEvent(
                tool_name="bash",
                args={"command": "echo 'test'"},
                tool_call_id="test-call-1",
                tool_class=BashTool,
            ),
            ToolResultEvent(
                tool_name="bash",
                tool_call_id="test-call-1",
                tool_class=BashTool,
                result={"output": "test"},
            ),
            AssistantEvent(content="Command executed successfully"),
        ],
    )
    engine.initialize()
    return engine


@pytest.fixture
def fake_engine_with_compact():
    """Create FakeVibeLangChainEngine for testing compact functionality."""
    from tests.stubs.fake_backend import FakeVibeLangChainEngine
    from vibe.core.types import AssistantEvent

    engine = FakeVibeLangChainEngine(
        config=langchain_config(),
        events_to_yield=[
            AssistantEvent(content="Message 1"),
            AssistantEvent(content="Message 2"),
            AssistantEvent(content="Message 3"),
            AssistantEvent(content="Message 4"),
        ],
    )
    engine.initialize()
    return engine

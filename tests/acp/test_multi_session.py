from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    InitializeRequest,
    NewSessionRequest,
    PromptRequest,
    RequestError,
)
from acp.schema import TextContentBlock
import pytest
from pytest import raises

from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_connection import FakeAgentSideConnection
from vibe.acp.acp_agent import VibeAcpAgent
from vibe.core.config import ModelConfig, VibeConfig
from vibe.core.types import Role


@pytest.fixture
def backend() -> FakeBackend:
    backend = FakeBackend()
    return backend


@pytest.fixture
def acp_agent(backend: FakeBackend) -> VibeAcpAgent:
    config = VibeConfig(
        active_model="devstral-latest",
        models=[
            ModelConfig(
                name="devstral-latest", provider="mistral", alias="devstral-latest"
            )
        ],
    )

    vibe_acp_agent: VibeAcpAgent | None = None

    def _create_agent(connection: Any) -> VibeAcpAgent:
        nonlocal vibe_acp_agent
        vibe_acp_agent = VibeAcpAgent(connection)
        return vibe_acp_agent

    FakeAgentSideConnection(_create_agent)
    return vibe_acp_agent  # pyright: ignore[reportReturnType]


class TestMultiSessionCore:
    @pytest.mark.skip(reason="ACP tests left aside for now - tracked separately")
    @pytest.mark.asyncio
    async def test_different_sessions_use_different_agents(
        self, acp_agent: VibeAcpAgent
    ) -> None:
        await acp_agent.initialize(InitializeRequest(protocolVersion=PROTOCOL_VERSION))
        session1_response = await acp_agent.newSession(
            NewSessionRequest(cwd=str(Path.cwd()), mcpServers=[])
        )
        session1 = acp_agent.sessions[session1_response.sessionId]
        session2_response = await acp_agent.newSession(
            NewSessionRequest(cwd=str(Path.cwd()), mcpServers=[])
        )
        session2 = acp_agent.sessions[session2_response.sessionId]

        assert session1.id != session2.id
        # Each agent should be independent
        assert session1.agent is not session2.agent
        assert id(session1.agent) != id(session2.agent)

    @pytest.mark.skip(reason="ACP tests left aside for now - tracked separately")
    @pytest.mark.asyncio
    async def test_error_on_nonexistent_session(self, acp_agent: VibeAcpAgent) -> None:
        await acp_agent.initialize(InitializeRequest(protocolVersion=PROTOCOL_VERSION))
        await acp_agent.newSession(
            NewSessionRequest(cwd=str(Path.cwd()), mcpServers=[])
        )

        fake_session_id = "fake-session-id-" + str(uuid4())

        with raises(RequestError) as exc_info:
            await acp_agent.prompt(
                PromptRequest(
                    sessionId=fake_session_id,
                    prompt=[TextContentBlock(type="text", text="Hello, world!")],
                )
            )

        assert isinstance(exc_info.value, RequestError)
        assert str(exc_info.value) == "Invalid params"

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="VibeLangChainEngine doesn't use FakeBackend in the same way as legacy Agent. "
        "The test tries to use FakeBackend._chunks but VibeLangChainEngine makes actual "
        "LangChain calls. To re-enable this test, it needs to be rewritten to use proper "
        "LangChain mocking (e.g., with langchain_core.messages.human.AIMessage). "
        "Tracked in separate issue for LangChain migration."
    )
    async def test_simultaneous_message_processing(
        self, acp_agent: VibeAcpAgent, backend: FakeBackend
    ) -> None:
        # Test skipped - see above reason
        pass

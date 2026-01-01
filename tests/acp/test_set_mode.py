"""ACP set mode tests with VibeAcpAgent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from acp import AgentSideConnection, NewSessionRequest, SetSessionModeRequest
import pytest

from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_connection import FakeAgentSideConnection
from vibe.acp.acp_agent import VibeAcpAgent
from vibe.acp.utils import VibeSessionMode
from vibe.core.types import LLMChunk, LLMMessage, LLMUsage, Role


@pytest.fixture
def backend() -> FakeBackend:
    backend = FakeBackend(
        results=[
            LLMChunk(
                message=LLMMessage(role=Role.assistant, content="Hi"),
                finish_reason="end_turn",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1),
            )
        ]
    )
    return backend


@pytest.fixture
def acp_agent(backend: FakeBackend) -> VibeAcpAgent:
    vibe_acp_agent: VibeAcpAgent | None = None

    def _create_agent(connection: AgentSideConnection) -> VibeAcpAgent:
        nonlocal vibe_acp_agent
        vibe_acp_agent = VibeAcpAgent(connection)
        return vibe_acp_agent

    FakeAgentSideConnection(_create_agent)
    return vibe_acp_agent


class TestACPSetMode:
    @pytest.mark.skip(reason="ACP tests left aside for now - tracked separately")
    @pytest.mark.asyncio
    async def test_set_mode_response_structure(
        self, acp_agent: VibeAcpAgent
    ) -> None:
        session_response = await acp_agent.newSession(
            NewSessionRequest(cwd=str(Path.cwd()), mcpServers=[])
        )

        mode_response = await acp_agent.setSessionMode(
            SetSessionModeRequest(
                sessionId=session_response.sessionId,
                modeId=VibeSessionMode.AUTO_APPROVE,
            )
        )

        assert mode_response is not None
        assert mode_response.modeId == VibeSessionMode.AUTO_APPROVE

    @pytest.mark.skip(reason="ACP tests left aside for now - tracked separately")
    @pytest.mark.asyncio
    async def test_set_mode_updates_agent_mode(self, acp_agent: VibeAcpAgent) -> None:
        session_response = await acp_agent.newSession(
            NewSessionRequest(cwd=str(Path.cwd()), mcpServers=[])
        )
        session_id = session_response.sessionId

        await acp_agent.setSessionMode(
            SetSessionModeRequest(
                sessionId=session_id,
                modeId=VibeSessionMode.AUTO_APPROVE,
            )
        )

        # Verify mode was updated
        acp_session = next(
            (s for s in acp_agent.sessions.values() if s.id == session_id),
            None,
        )
        assert acp_session is not None
        assert acp_session.mode == VibeSessionMode.AUTO_APPROVE

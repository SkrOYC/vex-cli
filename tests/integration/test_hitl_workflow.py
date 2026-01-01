"""Integration tests for HITL interrupt workflow."""

from __future__ import annotations

import pytest

from tests.mock.utils import (
    create_mock_hitl_request,
    create_mock_hitl_response,
)
from tests.stubs.fake_backend import FakeInterruptedAgent
from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_engine import VibeLangChainEngine


class TestHITLWorkflow:
    """Test complete HITL interrupt and approval workflow."""

    @pytest.mark.asyncio
    async def test_single_tool_approve(self, langchain_config: VibeConfig):
        """Test approving a single tool execution."""
        engine = VibeLangChainEngine(langchain_config)
        engine.initialize()

        # Mock agent with interrupt
        fake_agent = FakeInterruptedAgent()
        fake_agent.add_interrupt("bash", {"command": "ls"}, "Execute bash")
        engine._agent = fake_agent  # type: ignore[assignment]

        # Handle approval
        await engine.handle_approval(True, None)

        # Verify correct HITLResponse sent
        decisions = fake_agent.get_decisions()
        assert len(decisions) == 1
        assert decisions[0]["decisions"][0]["type"] == "approve"

    @pytest.mark.asyncio
    async def test_single_tool_reject(self, langchain_config: VibeConfig):
        """Test rejecting a single tool execution."""
        engine = VibeLangChainEngine(langchain_config)
        engine.initialize()

        fake_agent = FakeInterruptedAgent()
        fake_agent.add_interrupt("bash", {"command": "rm -rf /"}, "Dangerous command")
        engine._agent = fake_agent  # type: ignore[assignment]

        await engine.handle_approval(False, "Unsafe command")

        decisions = fake_agent.get_decisions()
        assert len(decisions) == 1
        assert decisions[0]["decisions"][0]["type"] == "reject"
        assert decisions[0]["decisions"][0]["message"] == "Unsafe command"

    @pytest.mark.asyncio
    async def test_single_tool_default_reject_message(
        self,
        langchain_config: VibeConfig,
    ):
        """Test rejecting with default message."""
        engine = VibeLangChainEngine(langchain_config)
        engine.initialize()

        fake_agent = FakeInterruptedAgent()
        fake_agent.add_interrupt("bash", {"command": "test"}, "Test")
        engine._agent = fake_agent  # type: ignore[assignment]

        await engine.handle_approval(False, None)

        decisions = fake_agent.get_decisions()
        assert len(decisions) == 1
        assert decisions[0]["decisions"][0]["type"] == "reject"
        message = decisions[0]["decisions"][0].get("message", "")
        assert "Operation rejected by user" in message

    @pytest.mark.asyncio
    async def test_multi_tool_approval(self, langchain_config: VibeConfig):
        """Test approving/rejecting multiple tools individually."""
        engine = VibeLangChainEngine(langchain_config)
        engine.initialize()

        fake_agent = FakeInterruptedAgent()
        fake_agent.add_interrupt("bash", {"command": "ls"}, "List files")
        fake_agent.add_interrupt("read_file", {"path": "test.txt"}, "Read file")
        fake_agent.add_interrupt("edit", {"path": "test.txt"}, "Edit file")
        engine._agent = fake_agent  # type: ignore[assignment]

        # Approve individually
        approvals = [True, False, True]
        feedbacks: list[str | None] = [None, "Unsafe", None]

        await engine.handle_multi_tool_approval(approvals, feedbacks)

        decisions = fake_agent.get_decisions()
        assert len(decisions) == 1
        assert len(decisions[0]["decisions"]) == 3
        assert decisions[0]["decisions"][0]["type"] == "approve"
        assert decisions[0]["decisions"][1]["type"] == "reject"
        assert decisions[0]["decisions"][2]["type"] == "approve"

    @pytest.mark.asyncio
    async def test_approve_all(self, langchain_config: VibeConfig):
        """Test approving all tools with batch shortcut."""
        engine = VibeLangChainEngine(langchain_config)
        engine.initialize()

        fake_agent = FakeInterruptedAgent()
        for i in range(5):
            fake_agent.add_interrupt(
                "bash",
                {"command": f"test{i}"},
                f"Test {i}",
            )
        engine._agent = fake_agent  # type: ignore[assignment]

        await engine.handle_approve_all(5)

        decisions = fake_agent.get_decisions()
        assert len(decisions) == 1
        assert len(decisions[0]["decisions"]) == 5
        assert all(d["type"] == "approve" for d in decisions[0]["decisions"])

    @pytest.mark.asyncio
    async def test_reject_all(self, langchain_config: VibeConfig):
        """Test rejecting all tools with batch shortcut."""
        engine = VibeLangChainEngine(langchain_config)
        engine.initialize()

        fake_agent = FakeInterruptedAgent()
        for i in range(3):
            fake_agent.add_interrupt(
                "bash",
                {"command": f"test{i}"},
                f"Test {i}",
            )
        engine._agent = fake_agent  # type: ignore[assignment]

        await engine.handle_reject_all(3, "Batch reject")

        decisions = fake_agent.get_decisions()
        assert len(decisions) == 1
        assert len(decisions[0]["decisions"]) == 3
        assert all(d["type"] == "reject" for d in decisions[0]["decisions"])
        assert all(d["message"] == "Batch reject" for d in decisions[0]["decisions"])

    @pytest.mark.asyncio
    async def test_length_mismatch(self, langchain_config: VibeConfig):
        """Test ValueError on length mismatch."""
        engine = VibeLangChainEngine(langchain_config)
        engine.initialize()

        fake_agent = FakeInterruptedAgent()
        fake_agent.add_interrupt("bash", {"command": "test"}, "Test")
        engine._agent = fake_agent  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Length mismatch"):
            await engine.handle_multi_tool_approval([True, False], [None])

    @pytest.mark.asyncio
    async def test_handle_approval_no_agent(self, langchain_config: VibeConfig):
        """Test handle_approval gracefully handles None agent."""
        engine = VibeLangChainEngine(langchain_config)

        # Should not raise even with no agent
        await engine.handle_approval(True, "test")
        await engine.handle_approval(False, "rejected")

    @pytest.mark.asyncio
    async def test_multi_tool_no_agent(self, langchain_config: VibeConfig):
        """Test multi-tool approval gracefully handles None agent."""
        engine = VibeLangChainEngine(langchain_config)

        # Should not raise
        await engine.handle_multi_tool_approval([True], [None])


class TestHITLMockUtilities:
    """Test mock utility functions for HITL."""

    def test_create_mock_hitl_request_defaults(self):
        """Test create_mock_hitl_request with defaults."""
        request = create_mock_hitl_request()

        assert len(request["action_requests"]) == 1
        assert request["action_requests"][0]["name"] == "bash"
        assert request["action_requests"][0]["args"] == {"command": "test"}
        assert request["action_requests"][0]["description"] == "Execute bash"

    def test_create_mock_hitl_request_custom(self):
        """Test create_mock_hitl_request with custom values."""
        request = create_mock_hitl_request(
            tool_name="read_file",
            args={"path": "/tmp/file.txt"},
            description="Read temp file",
            allowed_decisions=["approve"],
        )

        assert len(request["action_requests"]) == 1
        assert request["action_requests"][0]["name"] == "read_file"
        assert request["action_requests"][0]["args"] == {"path": "/tmp/file.txt"}
        assert request["action_requests"][0]["description"] == "Read temp file"
        assert request["review_configs"][0]["allowed_decisions"] == ["approve"]

    def test_create_mock_hitl_response_approve(self):
        """Test create_mock_hitl_response for approval."""
        response = create_mock_hitl_response(approved=True)

        assert len(response["decisions"]) == 1
        assert response["decisions"][0]["type"] == "approve"
        assert "message" not in response["decisions"][0]

    def test_create_mock_hitl_response_reject(self):
        """Test create_mock_hitl_response for rejection."""
        response = create_mock_hitl_response(approved=False, message="Not safe")

        assert len(response["decisions"]) == 1
        assert response["decisions"][0]["type"] == "reject"
        assert response["decisions"][0]["message"] == "Not safe"

    def test_create_mock_hitl_response_default_message(self):
        """Test create_mock_hitl_response with default message."""
        response = create_mock_hitl_response(approved=False)

        assert len(response["decisions"]) == 1
        assert response["decisions"][0]["type"] == "reject"
        assert response["decisions"][0]["message"] == "Rejected"


class TestFakeInterruptedAgent:
    """Test FakeInterruptedAgent mock utility."""

    def test_add_and_get_interrupt(self, fake_interrupted_agent: FakeInterruptedAgent):
        """Test adding and retrieving interrupts."""
        fake_interrupted_agent.add_interrupt(
            tool_name="bash",
            args={"command": "ls"},
            description="List files",
        )

        interrupt = fake_interrupted_agent.get_interrupt()

        assert interrupt is not None
        assert len(interrupt["action_requests"]) == 1
        assert interrupt["action_requests"][0]["name"] == "bash"
        assert interrupt["action_requests"][0]["description"] == "List files"

    def test_multiple_interrupts(self, fake_interrupted_agent: FakeInterruptedAgent):
        """Test adding and retrieving multiple interrupts."""
        fake_interrupted_agent.add_interrupt("bash", {"command": "ls"}, "List")
        fake_interrupted_agent.add_interrupt("read", {"path": "file.txt"}, "Read")
        fake_interrupted_agent.add_interrupt("edit", {"path": "file.txt"}, "Edit")

        interrupt1 = fake_interrupted_agent.get_interrupt()
        interrupt2 = fake_interrupted_agent.get_interrupt()
        interrupt3 = fake_interrupted_agent.get_interrupt()
        interrupt4 = fake_interrupted_agent.get_interrupt()

        assert interrupt1 is not None
        assert interrupt1["action_requests"][0]["name"] == "bash"

        assert interrupt2 is not None
        assert interrupt2["action_requests"][0]["name"] == "read"

        assert interrupt3 is not None
        assert interrupt3["action_requests"][0]["name"] == "edit"

        assert interrupt4 is None

    def test_resume_with_decision(self, fake_interrupted_agent: FakeInterruptedAgent):
        """Test resuming with a decision."""
        fake_interrupted_agent.add_interrupt("bash", {"command": "ls"}, "List")

        decision = create_mock_hitl_response(approved=True)
        fake_interrupted_agent.resume(decision)

        decisions = fake_interrupted_agent.get_decisions()

        assert len(decisions) == 1
        assert decisions[0]["decisions"][0]["type"] == "approve"

    def test_multiple_decisions(self, fake_interrupted_agent: FakeInterruptedAgent):
        """Test tracking multiple decisions."""
        fake_interrupted_agent.add_interrupt("bash", {"command": "ls"}, "List")
        fake_interrupted_agent.add_interrupt("read", {"path": "file.txt"}, "Read")

        fake_interrupted_agent.resume(create_mock_hitl_response(approved=True))
        fake_interrupted_agent.resume(
            create_mock_hitl_response(approved=False, message="Unsafe")
        )

        decisions = fake_interrupted_agent.get_decisions()

        assert len(decisions) == 2
        assert decisions[0]["decisions"][0]["type"] == "approve"
        assert decisions[1]["decisions"][0]["type"] == "reject"
        assert decisions[1]["decisions"][0]["message"] == "Unsafe"

    def test_get_ainvoke_call_count(self, fake_interrupted_agent: FakeInterruptedAgent):
        """Test getting ainvoke call count."""
        assert fake_interrupted_agent.get_ainvoke_call_count() == 0

        # The count is tracked but not incremented by this mock
        # This tests that the attribute exists and can be accessed
        count = fake_interrupted_agent.get_ainvoke_call_count()
        assert isinstance(count, int)

    def test_allowed_decisions_custom(
        self, fake_interrupted_agent: FakeInterruptedAgent
    ):
        """Test custom allowed_decisions."""
        fake_interrupted_agent.add_interrupt(
            tool_name="safe_tool",
            args={"arg": "value"},
            description="Safe operation",
            allowed_decisions=["approve"],
        )

        interrupt = fake_interrupted_agent.get_interrupt()

        assert interrupt is not None
        assert len(interrupt["review_configs"]) == 1
        assert interrupt["review_configs"][0]["allowed_decisions"] == ["approve"]

    def test_empty_decisions(self, fake_interrupted_agent: FakeInterruptedAgent):
        """Test getting decisions when none have been made."""
        decisions = fake_interrupted_agent.get_decisions()
        assert len(decisions) == 0

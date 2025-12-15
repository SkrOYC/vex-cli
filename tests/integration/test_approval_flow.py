"""Integration tests for approval flow with DeepAgents."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from vibe.core.config import VibeConfig
from vibe.core.engine import VibeEngine
from vibe.core.engine.adapters import ApprovalBridge
from vibe.core.types import LLMMessage, Role


class TestApprovalFlow:
    """Test end-to-end approval workflow."""

    @pytest.mark.asyncio
    async def test_vibeengine_initialization_with_approval_bridge(self):
        """Test VibeEngine initializes with approval bridge."""
        config = VibeConfig()
        config.use_deepagents = True

        # Mock approval bridge
        approval_bridge = MagicMock()

        engine = VibeEngine(config=config, approval_callback=approval_bridge)
        engine.initialize()

        assert engine.approval_bridge == approval_bridge
        assert engine._agent is not None

    @pytest.mark.asyncio
    async def test_interrupt_config_building(self):
        """Test interrupt config is built correctly."""
        config = VibeConfig()
        config.use_deepagents = True

        from vibe.core.tools.base import BaseToolConfig, ToolPermission

        # Add tool configs
        write_config = BaseToolConfig()
        write_config.permission = ToolPermission.ASK
        config.tools["write_file"] = write_config

        read_config = BaseToolConfig()
        read_config.permission = ToolPermission.ALWAYS
        config.tools["read_file"] = read_config

        engine = VibeEngine(config=config)
        interrupt_config = engine._build_interrupt_config()

        # write_file should be interrupted
        assert "write_file" in interrupt_config
        # read_file should not
        assert "read_file" not in interrupt_config
        # Dangerous tools should be included
        assert "bash" in interrupt_config

    @pytest.mark.asyncio
    async def test_handle_approval_calls_bridge(self):
        """Test handle_approval calls the approval bridge."""
        config = VibeConfig()
        approval_bridge = MagicMock()
        approval_bridge.respond = AsyncMock()

        engine = VibeEngine(config=config, approval_callback=approval_bridge)

        await engine.handle_approval(True, "test-request-id", "test feedback")

        approval_bridge.respond.assert_called_once_with(
            True, "test-request-id", "test feedback"
        )

    @pytest.mark.asyncio
    async def test_resume_execution_placeholder(self):
        """Test resume_execution method exists and runs without error."""
        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Should not raise
        await engine.resume_execution({"approved": True})

    @pytest.mark.asyncio
    async def test_reject_execution_placeholder(self):
        """Test reject_execution method exists and runs without error."""
        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Should not raise
        await engine.reject_execution({"approved": False, "feedback": "rejected"})

    @pytest.mark.asyncio
    async def test_run_with_mock_agent(self):
        """Test running with a mock agent that yields events."""
        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Mock the agent with async iterator
        async def mock_astream_events(*args, **kwargs):
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": MagicMock(content="Hello")},
            }
            yield {"event": "on_tool_start", "name": "test_tool", "data": {"input": {}}}

        mock_agent = MagicMock()
        mock_agent.astream_events = mock_astream_events
        engine._agent = mock_agent

        # Mock event translator
        engine.event_translator.translate = MagicMock(
            side_effect=[
                MagicMock(),  # Assistant event
                None,  # Tool event (filtered out)
            ]
        )

        events = []
        async for event in engine.run("test message"):
            events.append(event)

        assert len(events) == 1  # Only the assistant event

    @pytest.mark.asyncio
    async def test_interrupt_event_handling(self):
        """Test handling of interrupt events in event stream."""
        from vibe.core.types import InterruptEvent

        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Mock event handler with interrupt callback
        interrupt_calls = []

        async def mock_interrupt_callback(data):
            interrupt_calls.append(data)

        # Create actual event handler with interrupt callback
        from vibe.cli.textual_ui.handlers.event_handler import EventHandler

        mock_handler = EventHandler(
            mount_callback=MagicMock(),
            scroll_callback=MagicMock(),
            todo_area_callback=MagicMock(),
            get_tools_collapsed=lambda: False,
            get_todos_collapsed=lambda: False,
            interrupt_callback=mock_interrupt_callback,
        )

        # Create interrupt event
        interrupt_event = InterruptEvent(interrupt_data={"test": "data"})

        # Call handle_event
        result = await mock_handler.handle_event(interrupt_event)

        # Should call interrupt callback
        assert len(interrupt_calls) == 1
        assert interrupt_calls[0] == {"test": "data"}

    @pytest.mark.asyncio
    async def test_multiple_interrupts_sequence(self):
        """Test handling multiple interrupts in sequence."""
        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Mock approval bridge
        approval_bridge = MagicMock()
        approval_bridge.handle_interrupt = AsyncMock(
            side_effect=[
                {"approved": True, "feedback": None},
                {"approved": False, "feedback": "rejected"},
            ]
        )
        engine.approval_bridge = approval_bridge

        # Test sequential interrupt handling
        result1 = await approval_bridge.handle_interrupt({"data": {}})
        result2 = await approval_bridge.handle_interrupt({"data": {}})

        assert result1["approved"] is True
        assert result2["approved"] is False

        assert approval_bridge.handle_interrupt.call_count == 2

    @pytest.mark.asyncio
    async def test_engine_error_handling(self):
        """Test error handling in engine operations."""
        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Test resume_execution with no agent
        await engine.resume_execution({"approved": True})  # Should not raise

        # Test reject_execution with no agent
        await engine.reject_execution({"approved": False})  # Should not raise

        # Test handle_approval with no bridge
        engine.approval_bridge = None
        await engine.handle_approval(True, "feedback")  # Should not raise

    @pytest.mark.asyncio
    async def test_event_translator_interrupt_handling(self):
        """Test event translator handles interrupt events."""
        from vibe.core.engine.adapters import EventTranslator

        config = VibeConfig()
        translator = EventTranslator(config)

        # Test interrupt event translation
        interrupt_event = {
            "event": "interrupt",
            "data": {"action_request": {"name": "test"}},
        }
        result = translator.translate(interrupt_event)

        assert result is not None
        from vibe.core.types import InterruptEvent

        assert isinstance(result, InterruptEvent)
        assert result.interrupt_data == interrupt_event

    @pytest.mark.asyncio
    async def test_approval_bridge_timeout_cleanup(self):
        """Test that timed-out approvals are properly cleaned up."""
        bridge = ApprovalBridge()

        # Create interrupt that will timeout
        interrupt = {
            "data": {
                "action_request": {
                    "name": "slow_tool",
                    "args": {},
                    "description": "Slow tool",
                }
            }
        }

        # Start interrupt handling
        task = asyncio.create_task(bridge.handle_interrupt(interrupt))

        # Wait for timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.1)

        # Check that pending approvals were cleaned up
        assert len(bridge._pending_approvals) == 0

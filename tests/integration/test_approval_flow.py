"""Integration tests for approval flow with DeepAgents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vibe.core.config import VibeConfig
from vibe.core.engine import VibeEngine
from vibe.core.engine.adapters import ApprovalBridge


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
        """Test handle_approval calls to approval bridge."""
        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Mock the approval bridge on the engine instance
        mock_bridge = MagicMock(spec=ApprovalBridge)
        mock_bridge.respond = AsyncMock()
        engine.approval_bridge = mock_bridge

        await engine.handle_approval(True, "test-request-id", "test feedback")

        mock_bridge.respond.assert_called_once_with(
            True, "test-request-id", "test feedback"
        )

    @pytest.mark.asyncio
    async def test_resume_execution_no_agent(self):
        """Test resume_execution does nothing when no agent."""
        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Should not raise
        await engine.resume_execution({"approved": True})

    @pytest.mark.asyncio
    async def test_reject_execution_with_agent(self):
        """Test reject_execution updates agent state with rejection message."""
        config = VibeConfig()
        engine = VibeEngine(config=config)

        # Mock agent
        mock_agent = MagicMock()
        engine._agent = mock_agent
        engine._thread_id = "test-thread"

        await engine.reject_execution({
            "approved": False,
            "feedback": "Operation rejected",
        })

        # Should call update_state with rejection message
        mock_agent.update_state.assert_called_once()
        call_args = mock_agent.update_state.call_args
        assert call_args[0][0] == {
            "configurable": {"thread_id": "test-thread"}
        }  # config
        assert "messages" in call_args[0][1]  # state update
        assert len(call_args[0][1]["messages"]) == 1
        assert "Operation rejected" in call_args[0][1]["messages"][0].content
        assert call_args[1]["as_node"] == "human"  # as_node keyword argument

    @pytest.mark.asyncio
    async def test_reject_execution_no_agent(self):
        """Test reject_execution does nothing when no agent."""
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
        VibeEngine(config=config)

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
        await mock_handler.handle_event(interrupt_event)

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
    async def test_approval_bridge_basic_functionality(self):
        """Test basic ApprovalBridge functionality."""
        from vibe.core.config import VibeConfig

        config = VibeConfig()

        # Test with no callback (auto-approve mode)
        bridge = ApprovalBridge(config=config)

        interrupt = {
            "data": {
                "action_request": {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "description": "Test tool",
                }
            }
        }

        result = await bridge.handle_interrupt(interrupt)
        assert result["approved"] is True
        assert "no approval callback" in result["feedback"]

        # Check that pending approvals were cleaned up
        assert len(bridge._pending_approvals) == 0

    @pytest.mark.asyncio
    async def test_approval_bridge_with_pattern_permissions(self):
        """Test ApprovalBridge respects pattern-based permissions."""
        from vibe.core.config import VibeConfig
        from vibe.core.tools.base import BaseToolConfig, ToolPermission

        config = VibeConfig()

        # Set up tool with allowlist pattern
        write_config = BaseToolConfig()
        write_config.permission = ToolPermission.ASK
        write_config.allowlist = ["*.txt"]
        config.tools["write_file"] = write_config

        bridge = ApprovalBridge(config=config)

        # Test allowlist match
        interrupt_allow = {
            "data": {
                "action_request": {
                    "name": "write_file",
                    "args": {"path": "test.txt", "content": "hello"},
                    "description": "Write to text file",
                }
            }
        }

        result = await bridge.handle_interrupt(interrupt_allow)
        assert result["approved"] is True
        assert "allowlist" in result["feedback"]

        # Test denylist match
        write_config.denylist = ["*.exe"]
        interrupt_deny = {
            "data": {
                "action_request": {
                    "name": "write_file",
                    "args": {"path": "malware.exe", "content": "bad"},
                    "description": "Write to exe file",
                }
            }
        }

        result = await bridge.handle_interrupt(interrupt_deny)
        assert result["approved"] is False
        assert "denylist" in result["feedback"]

    @pytest.mark.asyncio
    async def test_vibeengine_with_approval_callback(self):
        """Test VibeEngine with approval callback integration."""
        from vibe.core.config import VibeConfig

        config = VibeConfig()
        config.use_deepagents = True

        # Mock approval callback
        approval_decisions = []

        async def mock_callback(action_request):
            decision = {"approved": True, "always_approve": False, "feedback": None}
            approval_decisions.append(decision)
            return decision

        engine = VibeEngine(config=config, approval_callback=mock_callback)
        engine.initialize()

        assert engine.approval_bridge is not None
        assert len(approval_decisions) == 0  # No decisions yet

    @pytest.mark.asyncio
    async def test_tui_interrupt_handler_integration(self):
        """Test TUI interrupt handler integration with ApprovalBridge."""
        from vibe.core.config import VibeConfig
        from vibe.core.engine.adapters import ApprovalBridge

        config = VibeConfig()

        # Create mock approval bridge
        approval_results = []

        async def mock_callback(action_request):
            result = {"approved": True, "always_approve": False, "feedback": None}
            approval_results.append(result)
            return result

        bridge = ApprovalBridge(config=config, approval_callback=mock_callback)

        # Test interrupt handling
        interrupt = {
            "data": {
                "action_request": {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "description": "Test tool execution",
                }
            }
        }

        result = await bridge.handle_interrupt(interrupt)

        assert result["approved"] is True
        assert len(approval_results) == 1
        assert approval_results[0]["approved"] is True


class TestLangChainNativeHITL:
    """Integration tests for native LangChain 1.2.0 Human-in-the-Loop middleware.

    These tests verify that the native HumanInTheLoopMiddleware replaces
    the ApprovalBridge adapter when using LangChain 1.2.0.

    See issue #39 for details.
    """

    @pytest.mark.asyncio
    async def test_vibelangchainengine_has_native_hitl_middleware(self):
        """Test that VibeLangChainEngine uses native HumanInTheLoopMiddleware."""
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        # Create config with permissions that require approval
        config = VibeConfig()
        config.use_langchain = True

        # Add a tool that requires approval
        from vibe.core.tools.base import BaseToolConfig, ToolPermission

        bash_config = BaseToolConfig()
        bash_config.permission = ToolPermission.ASK
        config.tools["bash"] = bash_config

        engine = VibeLangChainEngine(config=config)

        # Verify the middleware stack contains HumanInTheLoopMiddleware
        middleware_stack = engine._build_middleware_stack()
        hitl_middleware = [
            m for m in middleware_stack if isinstance(m, HumanInTheLoopMiddleware)
        ]

        assert len(hitl_middleware) == 1, (
            "HumanInTheLoopMiddleware should be in the stack"
        )
        assert hitl_middleware[0].interrupt_on is not None

    @pytest.mark.asyncio
    async def test_vibelangchainengine_without_approval_bridge(self):
        """Test that VibeLangChainEngine doesn't use ApprovalBridge for HITL."""
        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        config = VibeConfig()
        config.use_langchain = True

        engine = VibeLangChainEngine(config=config)

        # VibeLangChainEngine should not have an approval_bridge attribute
        # (unlike the DeepAgents VibeEngine)
        assert not hasattr(engine, "approval_bridge") or engine.approval_bridge is None

    @pytest.mark.asyncio
    async def test_middleware_stack_order_for_hitl(self):
        """Test that HITL middleware is last in the stack (after context/price)."""
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        from vibe.core.engine.langchain_engine import VibeLangChainEngine
        from vibe.core.engine.langchain_middleware import (
            ContextWarningMiddleware,
            PriceLimitMiddleware,
        )

        config = VibeConfig()
        config.use_langchain = True
        config.context_warnings = True
        config.max_price = 10.0

        # Add a tool that requires approval
        from vibe.core.tools.base import BaseToolConfig, ToolPermission

        bash_config = BaseToolConfig()
        bash_config.permission = ToolPermission.ASK
        config.tools["bash"] = bash_config

        engine = VibeLangChainEngine(config=config)
        middleware_stack = engine._build_middleware_stack()

        # HITL should be last
        assert isinstance(middleware_stack[-1], HumanInTheLoopMiddleware)

        # Context and price middleware should come before HITL
        context_middleware = [
            m for m in middleware_stack if isinstance(m, ContextWarningMiddleware)
        ]
        price_middleware = [
            m for m in middleware_stack if isinstance(m, PriceLimitMiddleware)
        ]

        if context_middleware:
            context_idx = middleware_stack.index(context_middleware[0])
            hitl_idx = middleware_stack.index(middleware_stack[-1])
            assert context_idx < hitl_idx, (
                "ContextWarningMiddleware should be before HITL"
            )

        if price_middleware:
            price_idx = middleware_stack.index(price_middleware[0])
            hitl_idx = middleware_stack.index(middleware_stack[-1])
            assert price_idx < hitl_idx, "PriceLimitMiddleware should be before HITL"

    @pytest.mark.asyncio
    async def test_native_hitl_interrupt_on_config(self):
        """Test that interrupt_on config is properly built for native HITL."""
        from vibe.core.engine.langchain_engine import VibeLangChainEngine
        from vibe.core.engine.permissions import build_interrupt_config

        config = VibeConfig()
        config.use_langchain = True

        # Add tools with different permissions
        from vibe.core.tools.base import BaseToolConfig, ToolPermission

        write_config = BaseToolConfig()
        write_config.permission = ToolPermission.ASK
        config.tools["write_file"] = write_config

        read_config = BaseToolConfig()
        read_config.permission = ToolPermission.ALWAYS
        config.tools["read_file"] = read_config

        # Build interrupt config
        interrupt_on = build_interrupt_config(config)

        # write_file should require approval
        assert "write_file" in interrupt_on
        # read_file should auto-approve (not in interrupt_on)
        assert "read_file" not in interrupt_on

        engine = VibeLangChainEngine(config=config)
        middleware_stack = engine._build_middleware_stack()

        # Find HITL middleware and verify it has the correct config
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        hitl = next(
            (m for m in middleware_stack if isinstance(m, HumanInTheLoopMiddleware)),
            None,
        )
        assert hitl is not None
        # The middleware may transform the config, but write_file should still require approval
        assert "write_file" in hitl.interrupt_on

    @pytest.mark.asyncio
    async def test_langchain_engine_with_no_approvals_configured(self):
        """Test middleware stack when no custom tools require approval.

        Note: Even with no custom approvals, dangerous tools (bash, write_file, etc.)
        are always added to interrupt_on by build_interrupt_config() for security.
        """
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        from vibe.core.engine.langchain_engine import VibeLangChainEngine

        config = VibeConfig()
        config.use_langchain = True

        # Add only safe tools that don't require approval
        from vibe.core.tools.base import BaseToolConfig, ToolPermission

        read_config = BaseToolConfig()
        read_config.permission = ToolPermission.ALWAYS
        config.tools["read_file"] = read_config

        engine = VibeLangChainEngine(config=config)
        middleware_stack = engine._build_middleware_stack()

        # HITL middleware should still be in the stack because dangerous tools
        # (bash, write_file, etc.) are always added by build_interrupt_config
        from vibe.core.engine.permissions import build_interrupt_config

        interrupt_on = build_interrupt_config(config)

        # Dangerous tools are always in interrupt_on
        assert len(interrupt_on) > 0, "Dangerous tools should always require approval"

        hitl_middleware = [
            m for m in middleware_stack if isinstance(m, HumanInTheLoopMiddleware)
        ]
        assert len(hitl_middleware) == 1, "HITL should be added for dangerous tools"
